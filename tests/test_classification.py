"""
tests.test_classification
==========================

Focused checks for the classification extension (churn project): task-aware
metrics, catalog filtering, the generalised tuning grid, champion direction, and
the retention simulator's accounting identities. These do not need the trained
model or network — they exercise the pure logic added alongside the regression
showcase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_proposal import model_catalog as mc
from src.model_training import metrics as m


def test_all_metrics_is_task_aware():
    y = [0, 1, 1, 0, 1]
    pred = [0, 1, 0, 0, 1]
    score = [0.2, 0.9, 0.4, 0.1, 0.8]

    reg = m.all_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert set(reg) == {"rmse", "mae", "r2", "mape"}

    clf = m.all_metrics(y, pred, score, task="classification")
    assert set(clf) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= clf["roc_auc"] <= 1.0


def test_metric_direction_helpers():
    # Lower RMSE wins; higher ROC-AUC wins.
    assert m.is_better("rmse", 10.0, 20.0)
    assert not m.is_better("rmse", 30.0, 20.0)
    assert m.is_better("roc_auc", 0.9, 0.8)
    assert m.worst_value("rmse") == float("inf")
    assert m.worst_value("roc_auc") == float("-inf")


def test_catalog_lists_classification_models():
    clf = mc.models_for_task("classification")
    assert set(clf) == {"LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"}
    # Every classification spec carries the task tag and exactly one tunable param.
    for name in clf:
        spec = mc.get_spec(name)
        assert spec.task == "classification"
        assert len(spec.search_space) == 1


def test_roc_auc_single_class_is_nan():
    # All-negative labels -> ROC-AUC undefined, returned as NaN not a crash.
    assert np.isnan(m.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# Multiclass support.
# ---------------------------------------------------------------------------
def test_multiclass_metrics_use_macro_averaging():
    y = ["a", "b", "c", "a", "b", "c"]
    pred = ["a", "b", "c", "a", "c", "c"]   # one mistake
    assert m.is_multiclass(y)
    scores = m.all_metrics(y, pred, task="classification")
    # Macro averaging must not crash on 3 classes and must reflect the error.
    assert 0.0 < scores["precision"] < 1.0
    assert 0.0 < scores["recall"] < 1.0
    assert scores["accuracy"] == pytest.approx(5 / 6)


def test_multiclass_roc_auc_one_vs_rest():
    y = [0, 1, 2, 0, 1, 2]
    # Confident, correct probability matrix (columns = classes 0,1,2).
    proba = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    assert m.roc_auc(y, proba) == pytest.approx(1.0)


def test_binary_roc_auc_accepts_two_column_matrix():
    y = [0, 1, 0, 1]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    assert m.roc_auc(y, proba) == pytest.approx(1.0)


def test_multiclass_solver_override_only_for_logistic_multiclass():
    from src.model_training.trainer import multiclass_overrides

    multi = pd.Series(["a", "b", "c", "a"])
    binary = pd.Series([0, 1, 0, 1])
    # liblinear cannot fit 3+ classes -> swap solver, but only in that case.
    assert multiclass_overrides("LogisticRegression", multi) == {"solver": "lbfgs"}
    assert multiclass_overrides("LogisticRegression", binary) == {}
    assert multiclass_overrides("RandomForestClassifier", multi) == {}


def test_retention_simulation_identities():
    from src.simulation import retention

    rng = np.random.default_rng(1)
    scored = pd.DataFrame(
        {"churn_probability": rng.uniform(0, 1, 300), "monthly_revenue": rng.uniform(20, 120, 300)}
    )
    r = retention.simulate(scored, threshold=0.5, discount=0.1, acceptance=0.4, horizon_months=12)
    assert 0 <= r.n_targeted <= r.n_customers
    assert 0.0 <= r.retained <= r.expected_churners
    assert abs(r.net_value - (r.revenue_saved - r.campaign_cost)) < 1e-6
    # Nobody above an impossible threshold -> zero cost and zero ROI (no div/0).
    empty = retention.simulate(scored, threshold=1.01, discount=0.1, acceptance=0.5)
    assert empty.n_targeted == 0 and empty.roi == 0.0
