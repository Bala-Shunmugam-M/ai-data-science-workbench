"""
src.trust.subject
=================

PURPOSE
-------
Load the champion model and one split into the single bundle of arrays every
trust module needs, exactly once.

WHY THIS EXISTS
---------------
Fairness, permutation importance, calibration and the model card all need the
same five things: the champion's joblib bundle, the raw split frame (for the
subgroup columns, which are *not* in the design matrix once one-hot encoding has
run), the design matrix, the encoded truth, and the predictions. The evaluator
builds those with private ``_predict``/``_score``/``_metrics`` helpers; copying
them into four more modules is how the housing and churn paths drifted apart the
first time. One loader, four consumers.

PIPELINE POSITION
-----------------
    evaluate (freezes the champion) -> [subject] -> fairness / importance /
                                                    calibration / model card
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import active
from config.paths import ENGINEERED_TEST_PATH, ENGINEERED_VALIDATION_PATH
from src.artifacts import model_registry
from src.model_training.design_matrix import (
    class_scores,
    encode_target,
    positive_class_label,
)
from src.utils.file_utils import load_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["ConfigDriftError", "TrustSubject", "load_subject", "SPLIT_PATHS"]

#: The splits this stage may read. ``test`` is the default and the right choice:
#: the champion is frozen before this stage runs, so reading test produces a
#: report rather than a selection. ``validation`` is offered only for debugging -
#: the model was tuned against it, so its numbers flatter the model.
SPLIT_PATHS: dict[str, Path] = {
    "test": ENGINEERED_TEST_PATH,
    "validation": ENGINEERED_VALIDATION_PATH,
}


@dataclass(frozen=True)
class TrustSubject:
    """The champion, one split, and everything derived from the pair."""

    champion: dict[str, Any]
    label: str
    bundle: dict[str, Any]
    split: str
    #: The engineered split as written to disk. Carries the *original* columns,
    #: so subgroup auditing can use ``ocean_proximity`` or ``gender`` rather than
    #: the one-hot dummies the estimator actually saw.
    frame: pd.DataFrame
    #: The post-preprocessor numeric matrix the estimator was fit on.
    matrix: pd.DataFrame
    y_true: pd.Series
    y_pred: np.ndarray
    #: Positive-class probabilities (binary), the full proba matrix (multiclass),
    #: or None for regression and for classifiers without ``predict_proba``.
    y_score: np.ndarray | None
    task: str
    target: str
    selection_metric: str
    #: The classes the ESTIMATOR was fitted on, in ``predict_proba`` column order.
    #: Empty for regression.
    model_classes: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Refuse a classification subject that cannot say what its classes are.

        Every task branch in this package keys off :attr:`n_classes`. If
        ``model_classes`` were allowed to stay empty, ``n_classes`` would be 0,
        ``is_binary`` would be False, and a binary model would be silently scored
        down the multiclass path - the same shape of bug as inferring arity from the
        split. Failing at construction makes the omission impossible to miss.
        """

        if self.task == "classification" and not self.model_classes:
            raise ValueError(
                "A classification TrustSubject needs model_classes (the fitted "
                "estimator's classes_, in predict_proba column order). Without it "
                "no task branch can be chosen correctly."
            )

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def n_classes(self) -> int:
        """How many classes the MODEL knows about (0 for regression).

        Deliberately the model's arity, not the split's. A three-class model whose
        test split happens to contain only two labels is still a three-class model,
        and ``metrics.roc_auc`` documents that state as reachable. Deriving arity
        from the split let a multiclass model past the binary guards in
        :mod:`src.trust.calibration`, which then read column 1 of a three-column
        probability matrix as if it were "the positive class".
        """

        return len(self.model_classes)

    @property
    def n_classes_in_split(self) -> int:
        """Distinct labels actually present in this split's truth.

        Reporting only. Never use it to decide which code path to take - see
        :attr:`n_classes`.
        """

        if not self.is_classification:
            return 0
        return int(len(np.unique(np.asarray(self.y_true))))

    @property
    def is_binary(self) -> bool:
        return self.is_classification and self.n_classes == 2

    @property
    def positive_label(self) -> Any | None:
        """The label whose probability :attr:`y_score` holds, for a binary model.

        Delegates to :func:`src.model_training.design_matrix.positive_class_label`
        so the probability column and the truth indicator can never be derived by
        two different rules.
        """

        if not self.is_binary:
            return None
        return positive_class_label(self.bundle["estimator"])


class ConfigDriftError(RuntimeError):
    """The active config no longer describes the model that was trained."""


def _check_config_matches_bundle(
    bundle: dict[str, Any], *, task: str, positive_class: str | None
) -> None:
    """Refuse to audit a model against a config that has changed under it.

    Each joblib bundle records the ``task`` and ``positive_class`` in force when it
    was trained. Nothing stops ``datasets.yaml`` being edited afterwards, and this
    stage is the one place that has both the bundle and the live config open.

    Swapping ``positive_class`` from ``Yes`` to ``No`` inverts ``y_true`` while the
    model is untouched: every fairness rate flips, the calibration curve mirrors,
    and permutation importance is measured against an inverted target. All of it
    reports cleanly. Failing loudly here is the only cheap defence.
    """

    mismatches = [
        f"{name}: model was trained with {trained!r}, config now says {current!r}"
        for name, trained, current in (
            ("task", bundle.get("task"), task),
            ("positive_class", bundle.get("positive_class"), positive_class),
        )
        # A bundle predating the field carries None; that is missing provenance,
        # not a conflict, so it is not treated as drift.
        if trained is not None and trained != current
    ]
    if mismatches:
        raise ConfigDriftError(
            "The active dataset config does not match the champion's training "
            "config, so every metric below would describe a different problem "
            "than the model solves:\n  - "
            + "\n  - ".join(mismatches)
            + "\nRetrain, or restore the config that produced this model."
        )


def load_subject(split: str = "test") -> TrustSubject | None:
    """
    Load the active project's champion against ``split``.

    Returns ``None`` when no champion is registered. That is a legitimate state -
    the trust stage running before ``evaluate`` - and the callers render it as
    "not available" rather than failing the pipeline. A *missing split file*, by
    contrast, raises: that means the pipeline is broken, not merely incomplete.
    """

    if split not in SPLIT_PATHS:
        raise ValueError(
            f"Unknown split '{split}'. Choose one of: {', '.join(sorted(SPLIT_PATHS))}."
        )

    champion = model_registry.get_champion()
    if champion is None:
        logger.info("No champion registered; trust artifacts need 'evaluate' first.")
        return None

    import joblib

    bundle = joblib.load(champion["model_path"])
    frame = load_csv(SPLIT_PATHS[split])

    target = active.target_column()
    task = active.task()
    positive_class = active.positive_class()
    _check_config_matches_bundle(bundle, task=task, positive_class=positive_class)
    y_true = encode_target(frame[target], positive_class)

    predictor_columns = bundle["predictor_columns"]
    matrix = bundle["preprocessor"].transform(frame[predictor_columns])
    estimator = bundle["estimator"]
    y_pred = estimator.predict(matrix)
    y_score = class_scores(estimator, matrix) if task == "classification" else None

    label = f"{bundle.get('name')} {bundle.get('version')}"
    logger.info("Trust subject: %s on the %s split (%d rows).", label, split, len(frame))

    return TrustSubject(
        champion=champion,
        label=label,
        bundle=bundle,
        split=split,
        frame=frame,
        matrix=matrix,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        task=task,
        target=target,
        selection_metric=active.selection_metric(),
        model_classes=(
            [c.item() if hasattr(c, "item") else c for c in estimator.classes_]
            if task == "classification" and hasattr(estimator, "classes_")
            else []
        ),
    )
