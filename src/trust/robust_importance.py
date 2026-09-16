"""
src.trust.robust_importance
===========================

PURPOSE
-------
Measure what the champion actually depends on, without asking the model to
grade its own homework.

WHY THIS EXISTS ALONGSIDE ``explainability/drivers.py``
------------------------------------------------------
``drivers.py`` reads ``coef_`` or ``feature_importances_`` - the model's own
internals. Those are cheap and, for a linear model, exact. They are also
model-specific and, for tree ensembles, biased: ``feature_importances_`` is
computed from training-set impurity decrease, which systematically inflates
high-cardinality and continuous features regardless of whether they help on
unseen data.

Permutation importance asks a different, harder question: shuffle one column of
held-out data and see how much the score degrades. A feature the model does not
really rely on costs nothing when destroyed. This works identically for a linear
model, a forest, or anything else with a ``predict``, and it measures dependence
on *unseen* data rather than on the training fit.

Both artifacts are worth having. When they disagree, the disagreement is the
finding - a feature with a large coefficient and no permutation importance is
one the model cannot actually use.

HOW TO READ THE NUMBERS
-----------------------
``importance_mean`` is the average drop in the selection metric when that column
is shuffled, over ``n_repeats`` shuffles. Bigger means the model depends on it
more. Values at or below zero mean no measurable dependence: shuffling the column
left the score unchanged or, by chance, improved it. Those are reported as
measured rather than clipped to zero, because a visibly negative value is
evidence of noise, and a clipped one silently invents a floor.

``importance_std`` is the spread across shuffles. An importance smaller than its
own standard deviation is not distinguishable from noise, which the
``distinguishable_from_noise`` flag records so a reader does not rank on it.

LIMITATIONS - both are real and neither is fixable here
------------------------------------------------------
1. Columns are permuted in the DESIGN matrix, which is post-one-hot. Shuffling
   one dummy column independently produces rows that encode no category or two at
   once - inputs the model could never see. The resulting importance is still
   informative about that specific level, but a categorical's total importance is
   not the sum of its dummies. This is the standard practice and the standard
   caveat; it is stated rather than hidden.
2. Correlated columns share credit unpredictably. When two inputs carry the same
   signal, shuffling either one alone barely moves the score, because the other
   still supplies it - so both look unimportant. Read the table as a group.

OUTPUTS
-------
    artifacts/trust/permutation_importance.json
    artifacts/trust/permutation_importance.csv
    artifacts/trust/figures/permutation_importance.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless: figures are written, never displayed.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from config.constants import RANDOM_STATE  # noqa: E402
from config.paths import (  # noqa: E402
    ROBUST_IMPORTANCE_CSV_PATH,
    ROBUST_IMPORTANCE_JSON_PATH,
    TRUST_FIGURES_DIR,
    ensure_dir,
)
from src.trust.subject import TrustSubject  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import save_csv, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

__all__ = ["DEFAULT_REPEATS", "MAX_ROWS", "scorer_for", "run"]

#: Ten shuffles per column: enough for the standard deviation to mean something,
#: cheap enough to run on every pipeline pass.
DEFAULT_REPEATS = 10

#: Permutation importance costs ``n_repeats * n_features`` full predictions. On a
#: large split that turns a fast stage into a slow one for no extra insight, so
#: rows beyond this are subsampled with the project seed. The sample size is
#: recorded in the artifact so the number is never read as if it used everything.
MAX_ROWS = 5_000

_PRECISION = 6

#: Project metric name -> scikit-learn scorer name. Permutation importance needs
#: a scorer where *higher is better*, which is why the error metrics map to their
#: ``neg_`` forms; the sign cancels out in the drop, so a positive importance
#: still means "shuffling this hurt".
_SCORERS: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "r2": "r2",
    "accuracy": "accuracy",
}

#: Metrics whose scorer name depends on how many classes there are.
_SCORERS_BY_ARITY: dict[str, tuple[str, str]] = {
    # (binary, multiclass)
    "precision": ("precision", "precision_macro"),
    "recall": ("recall", "recall_macro"),
    "f1": ("f1", "f1_macro"),
    "roc_auc": ("roc_auc", "roc_auc_ovr"),
}


def scorer_for(metric: str, *, is_binary: bool) -> str | None:
    """
    The scikit-learn scorer for a project selection metric, or ``None``.

    ``None`` means "no mapping exists", and the caller falls back to the
    estimator's own ``score`` method. Returning a wrong-but-plausible scorer
    would produce an importance table measured against a metric nobody chose.
    """

    if metric in _SCORERS:
        return _SCORERS[metric]
    if metric in _SCORERS_BY_ARITY:
        binary_name, multiclass_name = _SCORERS_BY_ARITY[metric]
        return binary_name if is_binary else multiclass_name
    return None


def _figure(rows: list[dict[str, Any]], champion: str, metric: str, top_n: int) -> Path:
    """Horizontal bars with a standard-deviation whisker per feature."""

    ensure_dir(TRUST_FIGURES_DIR)
    top = rows[:top_n][::-1]  # reversed so the largest sits at the top
    labels = [row["feature"] for row in top]
    values = [row["importance_mean"] for row in top]
    errors = [row["importance_std"] for row in top]
    # Grey out anything indistinguishable from noise so the chart cannot be read
    # as a ranking of features the measurement cannot actually separate.
    colors = [
        "#2a6f97" if row["distinguishable_from_noise"] else "#b8bcc0" for row in top
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(top) + 1.5)))
    ax.barh(labels, values, xerr=errors, color=colors, edgecolor="white", capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Drop in {metric} when the column is shuffled")
    ax.set_title(f"Permutation importance - {champion}")
    fig.tight_layout()
    path = TRUST_FIGURES_DIR / "permutation_importance.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run(
    subject: TrustSubject,
    *,
    n_repeats: int = DEFAULT_REPEATS,
    max_rows: int = MAX_ROWS,
    top_n: int = 15,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Compute permutation importance for the champion and write the artifacts.

    Always returns a payload. When the scorer cannot be resolved the run still
    happens against the estimator's default ``score``, and the artifact says
    which metric it used - an importance table is only interpretable next to the
    metric it was measured against.
    """

    matrix, y_true = subject.matrix, subject.y_true
    n_rows = int(len(matrix))
    sampled = n_rows > max_rows
    if sampled:
        # Seeded so a rerun produces the same table; the project's split and
        # tuning use the same seed.
        index = (
            pd.Series(range(n_rows))
            .sample(n=max_rows, random_state=random_state)
            .to_numpy()
        )
        matrix = matrix.iloc[index]
        y_true = y_true.iloc[index]

    # ``is_binary`` now describes the model, so a three-class estimator can no
    # longer be handed a binary scorer because its split held two labels - that
    # raised ValueError from sklearn and aborted the whole stage (including an
    # orchestrated `all`).
    scoring = scorer_for(subject.selection_metric, is_binary=subject.is_binary)
    # The binary precision/recall/f1 scorers default to ``pos_label=1``, which only
    # holds when the positive class really is labelled 1. A string target with
    # positive_class unset would crash inside sklearn with a message about
    # pos_label; fall back rather than abort the stage over a naming detail.
    if (
        scoring in {"precision", "recall", "f1"}
        and 1 not in (subject.model_classes or [])
    ):
        logger.info(
            "Selection metric '%s' maps to a scorer that assumes pos_label=1, but "
            "this model's classes are %s; using the estimator's default score() "
            "instead.",
            subject.selection_metric,
            subject.model_classes,
        )
        scoring = None
    measured_against = scoring or "the estimator's default score() method"
    if scoring is None:
        logger.info(
            "No scikit-learn scorer maps to selection metric '%s'; using the "
            "estimator's default score().",
            subject.selection_metric,
        )

    result = permutation_importance(
        subject.bundle["estimator"],
        matrix,
        y_true,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,  # ponytail: serial. Revisit only if this stage becomes the bottleneck.
    )

    rows: list[dict[str, Any]] = []
    for feature, mean, std in zip(
        matrix.columns, result.importances_mean, result.importances_std
    ):
        mean_value, std_value = float(mean), float(std)
        rows.append(
            {
                "feature": str(feature),
                "importance_mean": round(mean_value, _PRECISION),
                "importance_std": round(std_value, _PRECISION),
                # A drop smaller than the spread across its own shuffles is not
                # separable from chance. Reported, but flagged so nobody ranks on it.
                #
                # This is ~1 standard deviation of the per-shuffle spread, which at
                # n_repeats=10 is roughly 3x the standard error of the mean - so it
                # is materially STRICTER than a significance test and will grey out
                # some real dependencies. Deliberate: a false "this matters" is
                # costlier here than a false "cannot tell".
                "distinguishable_from_noise": bool(
                    math.isfinite(mean_value) and mean_value > std_value and mean_value > 0
                ),
            }
        )
    # NaN-safe: a bare key would let one NaN take an arbitrary rank and push a real
    # feature down, because every comparison against NaN is False. ``-inf`` sorts a
    # non-finite importance to the bottom, where "not measurable" belongs. Lesson #1
    # of docs/ML_PIPELINE_LESSONS.md; the mapped scorers make NaN hard to reach, so
    # this hardens the pattern rather than fixing an observed failure.
    rows.sort(
        key=lambda row: row["importance_mean"]
        if math.isfinite(row["importance_mean"])
        else float("-inf"),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    rows = [
        {
            key: row[key]
            for key in (
                "rank",
                "feature",
                "importance_mean",
                "importance_std",
                "distinguishable_from_noise",
            )
        }
        for row in rows
    ]

    figure = _figure(rows, subject.label, measured_against, top_n)

    payload: dict[str, Any] = {
        "artifact": "permutation_importance",
        "generated_at": utc_timestamp(),
        "champion": subject.label,
        "split": subject.split,
        "selection_metric": subject.selection_metric,
        "measured_against": measured_against,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "n_rows_available": n_rows,
        "n_rows_used": int(len(matrix)),
        "subsampled": sampled,
        "n_features": len(rows),
        "n_distinguishable_from_noise": sum(
            1 for row in rows if row["distinguishable_from_noise"]
        ),
        "importances": rows,
        "figure": str(figure),
        "interpretation_limits": [
            "Importance is the drop in the score above when one column is "
            "shuffled. It measures what the model relies on, not what causes the "
            "outcome.",
            "Columns are permuted in the post-one-hot design matrix, so a "
            "categorical's dummy columns are shuffled independently. A single "
            "dummy's importance is meaningful; a categorical's total importance "
            "is not the sum of its dummies.",
            "Correlated columns share credit. When two inputs carry the same "
            "signal, shuffling either alone barely moves the score and both look "
            "unimportant - read the table as groups, not as independent effects.",
            "Values at or below zero mean no measurable dependence. They are not "
            "clipped, because a negative value is useful evidence of how much "
            "noise this measurement carries.",
        ],
    }
    if sampled:
        payload["interpretation_limits"].append(
            f"Measured on a seeded random {len(matrix):,} of {n_rows:,} rows."
        )

    ensure_dir(ROBUST_IMPORTANCE_JSON_PATH.parent)
    save_json(payload, ROBUST_IMPORTANCE_JSON_PATH)
    save_csv(pd.DataFrame(rows), ROBUST_IMPORTANCE_CSV_PATH)

    logger.info(
        "Permutation importance: %d features, %d distinguishable from noise, "
        "measured against %s.",
        len(rows),
        payload["n_distinguishable_from_noise"],
        measured_against,
    )
    return payload
