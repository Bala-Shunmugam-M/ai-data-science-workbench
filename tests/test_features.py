"""Smoke tests for feature proposal and deterministic feature execution."""

from __future__ import annotations

import numpy as np

from src.feature_engineering.feature_proposal import build_proposal
from src.feature_engineering.transformers import (
    LOG_FEATURES,
    RATIO_FEATURES,
    apply_automatic_features,
)


def test_ratio_features_are_computed_correctly(synthetic_housing):
    result = apply_automatic_features(synthetic_housing, ["rooms_per_household"])
    expected = synthetic_housing["total_rooms"] / synthetic_housing["households"]
    np.testing.assert_allclose(result["rooms_per_household"], expected)


def test_log_features_use_log1p(synthetic_housing):
    result = apply_automatic_features(synthetic_housing, ["log_population"])
    expected = np.log1p(synthetic_housing["population"])
    np.testing.assert_allclose(result["log_population"], expected)


def test_safe_division_yields_nan_on_zero_denominator(synthetic_housing):
    frame = synthetic_housing.copy()
    frame.loc[frame.index[0], "households"] = 0
    result = apply_automatic_features(frame, ["rooms_per_household"])
    # Zero denominator -> NaN, never an infinity that would corrupt scaling.
    assert np.isnan(result.loc[frame.index[0], "rooms_per_household"])
    assert not np.isinf(result["rooms_per_household"]).any()


def test_proposal_marks_only_ratios_and_logs_automatic(synthetic_housing):
    proposal = build_proposal(synthetic_housing)
    automatic = {row["feature_name"] for row in proposal if row["automatic"]}
    assert automatic == set(RATIO_FEATURES) | set(LOG_FEATURES)

    # Scaling/encoding proposals exist but are not executed by this stage.
    optional = {row["feature_name"] for row in proposal if not row["automatic"]}
    assert "standardized_numeric_predictors" in optional
    assert "one_hot_categorical_predictors" in optional


def test_every_proposal_has_required_metadata(synthetic_housing):
    proposal = build_proposal(synthetic_housing)
    required = {
        "feature_name", "feature_group", "source_columns", "rationale",
        "automatic", "validation_required", "implementing_script",
    }
    for row in proposal:
        assert required.issubset(row.keys())
