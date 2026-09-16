"""
src.model_training.trainer
==========================

PURPOSE
-------
Train every APPROVED model on the engineered training split, tuning Ridge/Lasso
alpha on the dedicated validation split (never k-fold over train+validation, per
the professor's doctrine). Each fitted model is versioned, saved with metadata,
registered, and audit/experiment-logged.

GOVERNANCE GUARDS
-----------------
* Refuses to train any model not present in ``model_approval.json``
  (:func:`governance.approvals.require_model_approved` -> ``GovernanceError``).
* The modelling matrix is built only from the governed approved predictor
  columns, so the target and any non-approved column are excluded by
  construction.

PIPELINE POSITION
-----------------
    approval -> [training] -> evaluation

OUTPUTS
-------
    models/<name>/<version>/model.joblib
    models/<name>/<version>/metadata.json
    models/model_registry.json   (via src.artifacts.model_registry)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import active
from config.constants import RANDOM_STATE
from config.paths import (
    ENGINEERED_TRAIN_PATH,
    ENGINEERED_VALIDATION_PATH,
    MODELS_DIR,
    ensure_dir,
)
from src.artifacts import experiment_tracker, model_registry
from src.governance import approvals, audit_logger, lineage_tracker
from src.model_proposal.model_catalog import get_spec
from src.model_training import metrics as metric_helpers
from src.model_training.design_matrix import (
    ModelingPreprocessor,
    class_scores,
    encode_target,
    split_x_y,
)
from src.model_training.factory import build_estimator
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _param_grid(spec) -> tuple[str | None, list[Any]]:
    """
    Build the tuning grid for a spec's single tunable hyperparameter.

    Returns ``(param_name, values)``; ``(None, [])`` when the model has no
    tunable hyperparameter (e.g. plain LinearRegression). Supports ``log``
    (log-spaced floats), ``int`` (inclusive integer range), and ``fixed``.
    """

    if not spec.search_space:
        return None, []
    name, space = next(iter(spec.search_space.items()))
    if space.scale == "log":
        values = [float(v) for v in np.logspace(np.log10(space.low), np.log10(space.high), space.num)]
    elif space.scale == "int":
        values = list(range(int(space.low), int(space.high) + 1))
    else:  # "fixed"
        values = [space.low]
    return name, values


def _val_metric(
    estimator: Any,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
    metric: str,
) -> float:
    """Score a fitted estimator on the validation split for ``metric``."""

    if task == "classification":
        scores = metric_helpers.all_metrics(
            y_val, estimator.predict(x_val), class_scores(estimator, x_val),
            task="classification",
        )
    else:
        scores = metric_helpers.all_metrics(y_val, estimator.predict(x_val))
    return scores[metric]


def multiclass_overrides(name: str, y: pd.Series) -> dict[str, Any]:
    """
    Estimator params that must change when the target has 3+ classes.

    The catalogue pins LogisticRegression to ``liblinear`` (chosen for the binary
    churn problem), but that solver cannot fit multiclass. Switching to
    ``lbfgs`` only in the multiclass case keeps every binary result identical.
    """

    if name == "LogisticRegression" and metric_helpers.is_multiclass(y):
        return {"solver": "lbfgs"}
    return {}


def _tune(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    task: str,
    selection_metric: str,
    extra_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Select the model's tunable hyperparameter by validation ``selection_metric``.

    Returns ``(best_params, history)``. The direction of improvement is taken
    from :data:`metrics.METRIC_HIGHER_IS_BETTER`, so one loop serves both tasks.
    The test split is never touched here.
    """

    spec = get_spec(name)
    param, grid = _param_grid(spec)
    if param is None or not grid:
        return {}, []

    extra = extra_params or {}
    history: list[dict[str, Any]] = []
    best_value: Any = None
    best_score = metric_helpers.worst_value(selection_metric)
    for value in grid:
        candidate = build_estimator(name, **{param: value, **extra})
        candidate.fit(x_train, y_train)
        score = _val_metric(candidate, x_val, y_val, task, selection_metric)
        history.append({param: value, f"validation_{selection_metric}": score})
        if best_value is None or metric_helpers.is_better(selection_metric, score, best_score):
            best_score, best_value = score, value

    logger.info(
        "%s tuned: %s=%s (validation %s=%.4f over %d candidates)",
        name, param, best_value, selection_metric, best_score, len(grid),
    )
    return {param: best_value}, history


def train_model(
    name: str,
    *,
    preprocessor: ModelingPreprocessor,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    predictor_columns: list[str],
    data_hash: str,
    actor: str | None = None,
) -> dict[str, Any]:
    """
    Train, tune, persist, and register a single approved model. Returns its
    registry entry.
    """

    # GUARD: never train an unapproved model.
    approvals.require_model_approved(name)

    task = active.task()
    selection_metric = active.selection_metric()
    positive_class = active.positive_class()

    x_train, y_train = split_x_y(train_df, predictor_columns)
    x_val, y_val = split_x_y(validation_df, predictor_columns)
    # Classification labels are encoded to 0/1 (positive class -> 1); regression
    # targets pass through unchanged.
    y_train = encode_target(y_train, positive_class)
    y_val = encode_target(y_val, positive_class)
    x_train_m = preprocessor.transform(x_train)
    x_val_m = preprocessor.transform(x_val)

    started = time.perf_counter()
    extra_params = multiclass_overrides(name, y_train) if task == "classification" else {}
    best_params, tuning_history = _tune(
        name, x_train_m, y_train, x_val_m, y_val,
        task=task, selection_metric=selection_metric, extra_params=extra_params,
    )
    estimator = build_estimator(name, **{**best_params, **extra_params})
    estimator.fit(x_train_m, y_train)
    training_seconds = round(time.perf_counter() - started, 4)

    def _metrics(y_true: pd.Series, matrix: pd.DataFrame) -> dict[str, float]:
        if task == "classification":
            return metric_helpers.all_metrics(
                y_true, estimator.predict(matrix), class_scores(estimator, matrix),
                task="classification",
            )
        return metric_helpers.all_metrics(y_true, estimator.predict(matrix))

    train_metrics = _metrics(y_train, x_train_m)
    validation_metrics = _metrics(y_val, x_val_m)

    version = model_registry.next_version(name)
    model_dir = ensure_dir(MODELS_DIR / name / version)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    bundle = {
        "name": name,
        "version": version,
        "estimator": estimator,
        "preprocessor": preprocessor,
        "predictor_columns": predictor_columns,
        "feature_names": preprocessor.feature_names_,
        "task": task,
        "positive_class": positive_class,
    }
    joblib.dump(bundle, model_path)

    metadata: dict[str, Any] = {
        "name": name,
        "version": version,
        "estimator_path": get_spec(name).estimator_path,
        "params": {**get_spec(name).default_params, **extra_params, **best_params},
        "tuned": bool(best_params),
        "best_params": best_params,
        "tuning_history": tuning_history,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_seconds": training_seconds,
        "n_train_rows": int(len(train_df)),
        "n_features": len(preprocessor.feature_names_),
        "predictor_columns": predictor_columns,
        "data_hash": data_hash,
        "random_state": RANDOM_STATE,
        "created_at": utc_timestamp(),
        "model_path": str(model_path),
    }
    save_json(metadata, metadata_path)

    entry = model_registry.register_model(
        {
            "name": name,
            "version": version,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "params": metadata["params"],
            "validation_metrics": validation_metrics,
            "train_metrics": train_metrics,
            "data_hash": data_hash,
            "is_champion": False,
        }
    )

    experiment_tracker.log_experiment(
        {
            "model": name,
            "version": version,
            "params": metadata["params"],
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "duration_seconds": training_seconds,
            "data_hash": data_hash,
        }
    )
    audit_logger.log_event(
        "model_training_run",
        {
            "model": name,
            "version": version,
            f"validation_{selection_metric}": validation_metrics.get(selection_metric),
            "best_params": best_params,
            "model_path": str(model_path),
        },
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )

    val_summary = " ".join(
        f"{k}={v:.4f}" for k, v in validation_metrics.items()
    )
    logger.info(
        "Trained %s %s | val %s | %.3fs",
        name, version, val_summary, training_seconds,
    )
    return entry


def train_all(
    *,
    train_path: Path = ENGINEERED_TRAIN_PATH,
    validation_path: Path = ENGINEERED_VALIDATION_PATH,
    actor: str | None = None,
) -> list[dict[str, Any]]:
    """
    Train every approved model on the engineered splits. Returns registry entries.

    Raises ``GovernanceError`` (via the per-model guard) if no models are
    approved, so the approval gate is unmissable.
    """

    names = approvals.approved_models()
    if not names:
        # Force the guard's clear error message.
        approvals.require_model_approved("<none>")

    predictor_columns = approvals.approved_predictor_columns()
    train_df = load_csv(train_path)
    validation_df = load_csv(validation_path)
    data_hash = lineage_tracker.file_sha256(train_path) or ""

    # One preprocessor fit on train is shared across all models (identical matrix).
    # Categorical columns come from the active project's config (housing:
    # ocean_proximity; churn: its 16 nominal fields), so the matrix is correct
    # for either task.
    x_train, _ = split_x_y(train_df, predictor_columns)
    preprocessor = ModelingPreprocessor(
        predictor_columns=predictor_columns,
        categorical_names=active.categorical_columns(),
    ).fit(x_train)

    entries: list[dict[str, Any]] = []
    for name in names:
        entries.append(
            train_model(
                name,
                preprocessor=preprocessor,
                train_df=train_df,
                validation_df=validation_df,
                predictor_columns=predictor_columns,
                data_hash=data_hash,
                actor=actor,
            )
        )

    lineage_tracker.record_node(
        "model_training",
        input_paths=[train_path, validation_path],
        output_paths=[Path(e["model_path"]) for e in entries],
        script="src/model_training/trainer.py",
        params={"models": names, "predictor_columns": predictor_columns},
        extra={"trained_models": [f"{e['name']} {e['version']}" for e in entries]},
    )
    logger.info("Trained %d approved model(s).", len(entries))
    return entries
