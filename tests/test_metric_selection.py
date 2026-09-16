"""
tests.test_metric_selection
===========================

PURPOSE
-------
Degenerate metric values must never win a model selection.

Several metrics here return NaN *by design* when they are undefined:
``roc_auc`` when a class is absent from a split, ``mape`` when an actual value
is zero. Every comparison against NaN is ``False``, so naive ``max``/``min``
and ``>``/``<`` both go wrong in ways that produce a plausible champion and a
plausible number - no exception, no warning, just the wrong model promoted.

This module pins the two selection primitives and the ordering behaviour that
made the original bug invisible. The failure was ordering-dependent: a NaN in
the middle of the list selected correctly, a NaN first did not, which is why it
survived every earlier test run.
"""

from __future__ import annotations

import math

import pytest

from src.model_evaluation.evaluator import select_champion
from src.model_training import metrics as metric_helpers

NAN = float("nan")


# ---------------------------------------------------------------------------
# is_better - used by the hyperparameter search in trainer.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["roc_auc", "r2", "accuracy", "f1"])
def test_real_score_beats_undefined_incumbent_higher_is_better(metric: str) -> None:
    """A NaN incumbent must be displaceable, or the first candidate wins forever."""

    assert metric_helpers.is_better(metric, 0.95, NAN) is True


@pytest.mark.parametrize("metric", ["rmse", "mae", "mape"])
def test_real_score_beats_undefined_incumbent_lower_is_better(metric: str) -> None:
    assert metric_helpers.is_better(metric, 100.0, NAN) is True


@pytest.mark.parametrize("metric", ["roc_auc", "rmse"])
def test_undefined_candidate_never_wins(metric: str) -> None:
    assert metric_helpers.is_better(metric, NAN, 0.5) is False


def test_two_undefined_scores_do_not_swap() -> None:
    """Neither is better, so the incumbent stands."""

    assert metric_helpers.is_better("roc_auc", NAN, NAN) is False


def test_is_better_still_respects_metric_direction() -> None:
    """The NaN handling must not disturb ordinary comparisons."""

    assert metric_helpers.is_better("roc_auc", 0.9, 0.8) is True
    assert metric_helpers.is_better("roc_auc", 0.7, 0.8) is False
    assert metric_helpers.is_better("rmse", 100.0, 200.0) is True
    assert metric_helpers.is_better("rmse", 300.0, 200.0) is False


def test_none_is_treated_as_undefined() -> None:
    assert metric_helpers.is_better("roc_auc", None, 0.5) is False
    assert metric_helpers.is_better("roc_auc", 0.5, None) is True


# ---------------------------------------------------------------------------
# The champion-selection key in model_evaluation.evaluator
# ---------------------------------------------------------------------------


def _champion(rows: list[dict], metric: str) -> str:
    """The real selection function, not a copy of it.

    ``select_champion`` was extracted from inside ``evaluator.run`` precisely
    so this test binds to the shipping code. A mirrored implementation here
    would pass forever while the real one regressed.
    """

    return select_champion(rows, metric)["name"]


def _rows(scores: list[float], metric: str) -> list[dict]:
    return [
        {"name": f"m{i}", "validation_metrics": {metric: score}}
        for i, score in enumerate(scores)
    ]


def test_nan_first_does_not_win_champion() -> None:
    """The exact shape of the original bug: NaN in position 0."""

    assert _champion(_rows([NAN, 0.55, 0.95], "roc_auc"), "roc_auc") == "m2"


def test_nan_last_does_not_win_champion() -> None:
    assert _champion(_rows([0.55, 0.95, NAN], "roc_auc"), "roc_auc") == "m1"


def test_nan_middle_does_not_win_champion() -> None:
    """Passed even before the fix - included so the asymmetry stays visible."""

    assert _champion(_rows([0.55, NAN, 0.95], "roc_auc"), "roc_auc") == "m2"


def test_champion_selection_lower_is_better_with_nan() -> None:
    assert _champion(_rows([NAN, 200.0, 100.0], "rmse"), "rmse") == "m2"


def test_all_scores_undefined_still_returns_a_champion() -> None:
    """Every candidate NaN is a real outcome; it must not raise."""

    assert _champion(_rows([NAN, NAN], "roc_auc"), "roc_auc") in {"m0", "m1"}


def test_missing_metric_key_is_treated_as_undefined() -> None:
    rows = [
        {"name": "no-metric", "validation_metrics": {}},
        {"name": "scored", "validation_metrics": {"roc_auc": 0.7}},
    ]
    assert _champion(rows, "roc_auc") == "scored"


# ---------------------------------------------------------------------------
# The metrics that produce NaN in the first place
# ---------------------------------------------------------------------------


def test_roc_auc_returns_nan_for_a_single_class_split() -> None:
    """Documents the source of the NaN the selection code must survive."""

    assert math.isnan(metric_helpers.roc_auc([1, 1, 1], [0.2, 0.6, 0.9]))


def test_worst_value_direction() -> None:
    assert metric_helpers.worst_value("roc_auc") == float("-inf")
    assert metric_helpers.worst_value("rmse") == float("inf")
