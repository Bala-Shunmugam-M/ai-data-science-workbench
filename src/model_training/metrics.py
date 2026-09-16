"""
src.model_training.metrics
==========================

PURPOSE
-------
Regression metric helpers (RMSE, MAE, R2, MAPE) shared by the training and
evaluation stages so the numbers reported everywhere are computed identically.

PIPELINE POSITION
-----------------
Used by the trainer (train/validation metrics) and the evaluator (validation
comparison + champion test metrics).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error (same units as the target)."""

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""

    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination (R-squared)."""

    return float(r2_score(y_true, y_pred))


def mape(y_true: Any, y_pred: Any) -> float:
    """
    Mean absolute percentage error (%), ignoring zero-valued targets.

    Zero targets are excluded to avoid division by zero; when every target is
    zero the result is NaN.
    """

    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    nonzero = y_true_arr != 0
    if not nonzero.any():
        return float("nan")
    errors = np.abs(
        (y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]
    )
    return float(np.mean(errors) * 100.0)


# ---------------------------------------------------------------------------
# Classification metrics (added for the churn project). ``y_score`` is the
# predicted probability of the positive class, required for ROC-AUC.
# ---------------------------------------------------------------------------
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def is_multiclass(y_true: Any) -> bool:
    """True when the target carries more than two distinct labels."""

    return len(np.unique(np.asarray(y_true))) > 2


def _average_for(y_true: Any) -> str:
    """``"macro"`` for multiclass (every class weighs equally), else binary."""

    return "macro" if is_multiclass(y_true) else "binary"


def accuracy(y_true: Any, y_pred: Any) -> float:
    return float(accuracy_score(y_true, y_pred))


def precision(y_true: Any, y_pred: Any) -> float:
    """Precision — positive class for binary, macro-averaged for multiclass."""

    return float(
        precision_score(y_true, y_pred, average=_average_for(y_true), zero_division=0)
    )


def recall(y_true: Any, y_pred: Any) -> float:
    """Recall — positive class for binary, macro-averaged for multiclass."""

    return float(
        recall_score(y_true, y_pred, average=_average_for(y_true), zero_division=0)
    )


def f1(y_true: Any, y_pred: Any) -> float:
    """F1 — positive class for binary, macro-averaged for multiclass."""

    return float(f1_score(y_true, y_pred, average=_average_for(y_true), zero_division=0))


def roc_auc(y_true: Any, y_score: Any) -> float:
    """
    ROC-AUC. Binary takes positive-class scores; multiclass takes the full
    ``predict_proba`` matrix and is scored one-vs-rest, macro-averaged.

    Returns NaN when it is undefined (a single class present, or a class absent
    from this split so one-vs-rest cannot be computed).
    """

    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        return float("nan")
    score = np.asarray(y_score)
    try:
        if score.ndim > 1 and score.shape[1] > 2:
            return float(
                roc_auc_score(y_true_arr, score, multi_class="ovr", average="macro")
            )
        if score.ndim > 1:  # binary handed a 2-column matrix -> positive column
            score = score[:, 1]
        return float(roc_auc_score(y_true_arr, score))
    except ValueError:
        # e.g. a class present in training but absent from this split.
        return float("nan")


# ---------------------------------------------------------------------------
# Direction of improvement for each selection metric: True = higher is better.
# The trainer's tuning loop and the evaluator's champion pick consult this so
# one code path serves both tasks.
# ---------------------------------------------------------------------------
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "rmse": False,
    "mae": False,
    "mape": False,
    "r2": True,
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
}


def is_better(metric: str, candidate: float, incumbent: float) -> bool:
    """Return True if ``candidate`` beats ``incumbent`` for ``metric``.

    NaN is handled explicitly, because several metrics here return it by design
    when they are undefined (``roc_auc`` with a class absent from the split,
    ``mape`` with a zero actual). Every comparison against NaN is ``False``, so
    a plain ``>``/``<`` gives two wrong answers: a NaN incumbent can never be
    displaced by a real score, and the search silently keeps whichever
    candidate happened to be evaluated first.

    An undefined score is treated as no score at all: it never wins, and it
    always loses to a real one.
    """

    candidate_undefined = candidate is None or (
        isinstance(candidate, float) and math.isnan(candidate)
    )
    incumbent_undefined = incumbent is None or (
        isinstance(incumbent, float) and math.isnan(incumbent)
    )

    if candidate_undefined:
        return False
    if incumbent_undefined:
        return True

    higher = METRIC_HIGHER_IS_BETTER.get(metric, False)
    return candidate > incumbent if higher else candidate < incumbent


def worst_value(metric: str) -> float:
    """The worst possible starting value for a champion search on ``metric``."""

    return float("-inf") if METRIC_HIGHER_IS_BETTER.get(metric, False) else float("inf")


def all_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any = None,
    task: str = "regression",
) -> dict[str, float]:
    """
    Return every metric for ``task`` as a JSON-serialisable dict.

    Regression (default, 2-arg call) is unchanged. Classification additionally
    uses ``y_score`` (positive-class probabilities) for ROC-AUC.
    """

    if task == "classification":
        metrics = {
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1": f1(y_true, y_pred),
        }
        if y_score is not None:
            metrics["roc_auc"] = roc_auc(y_true, y_score)
        return metrics

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
