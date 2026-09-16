"""
tests.test_automl
=================

Checks for the bring-your-own-dataset auto-detection heuristics: task inference,
target guessing, positive-class choice, id/high-cardinality dropping. Pure logic
on synthetic frames — no upload, model, or network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.automl import detect as d


def _rng():
    return np.random.default_rng(0)


def test_detects_regression_target():
    df = pd.DataFrame({"x1": _rng().normal(size=200), "x2": _rng().normal(size=200),
                       "price": _rng().normal(size=200) * 1000})
    det = d.detect(df, target="price")
    assert det.task == "regression"
    assert det.selection_metric == "rmse"
    assert det.positive_class is None


def test_detects_binary_classification_and_positive_class():
    n = 300
    df = pd.DataFrame({
        "f1": _rng().normal(size=n),
        "plan": _rng().choice(["A", "B", "C"], size=n),
        "Churn": _rng().choice(["Yes", "No"], size=n, p=[0.25, 0.75]),
    })
    det = d.detect(df, target="Churn")
    assert det.task == "classification"
    assert det.n_classes == 2
    assert det.positive_class == "Yes"          # Yes-token wins
    assert "plan" in det.categorical_columns     # object predictor -> categorical
    assert det.selection_metric == "roc_auc"


def test_detects_multiclass_as_trainable_classification():
    df = pd.DataFrame({"f": range(90), "species": ["a", "b", "c"] * 30})
    det = d.detect(df, target="species")
    assert det.task == "classification"      # trainable, not blocked
    assert det.n_classes == 3
    assert det.is_multiclass
    assert det.positive_class is None        # no single positive class when 3+
    assert det.selection_metric == "roc_auc"


def test_id_heuristic_does_not_eat_features_ending_in_id():
    """Regression test: 'malic_acid'/'valid'/'humid' are features, not identifiers."""

    n = 60
    df = pd.DataFrame({
        "malic_acid": _rng().normal(size=n),
        "humid": _rng().normal(size=n),
        "is_valid": _rng().integers(0, 2, size=n),
        "customerID": [f"c{i}" for i in range(n)],
        "target": _rng().integers(0, 2, size=n),
    })
    det = d.detect(df, target="target")
    assert "customerID" in det.drop_columns          # a real identifier
    for feature in ("malic_acid", "humid", "is_valid"):
        assert feature not in det.drop_columns, f"{feature} was wrongly dropped"


def test_name_tokenizer_splits_camel_and_snake():
    assert d._name_tokens("customerID") == ["customer", "id"]
    assert d._name_tokens("customer_id") == ["customer", "id"]
    assert d._name_tokens("malic_acid") == ["malic", "acid"]


def test_drops_id_and_highcard_columns():
    n = 120
    df = pd.DataFrame({
        "customerID": [f"id-{i}" for i in range(n)],   # all-unique identifier
        "notes": [f"free text {i}" for i in range(n)],  # high-cardinality text
        "amount": _rng().normal(size=n),
        "label": _rng().integers(0, 2, size=n),
    })
    det = d.detect(df, target="label")
    assert "customerID" in det.drop_columns
    assert "notes" in det.drop_columns
    assert "amount" not in det.drop_columns
    assert det.task == "classification"


def test_guess_target_prefers_name_hint():
    df = pd.DataFrame({"a": [1, 2, 3], "outcome": [0, 1, 0], "b": [4, 5, 6]})
    assert d.guess_target(df) == "outcome"
