"""Smoke tests for the semantic schema integrity and the validator."""

from __future__ import annotations

from src.data_manager.data_validator import status_from_issues, validate_data
from src.domain.schema import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    TARGET_COLUMN,
    schema_columns,
)


def test_target_and_groups_are_consistent():
    assert TARGET_COLUMN == "median_house_value"
    assert TARGET_COLUMN not in NUMERICAL_COLUMNS
    assert TARGET_COLUMN not in CATEGORICAL_COLUMNS
    assert CATEGORICAL_COLUMNS == ["ocean_proximity"]
    # Every numerical/categorical column exists in the schema.
    for column in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS:
        assert column in SEMANTIC_SCHEMA


def test_schema_has_expected_columns():
    expected = {
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "median_house_value", "ocean_proximity",
    }
    assert set(schema_columns()) == expected


def test_only_total_bedrooms_is_nullable():
    nullable = [c for c, r in SEMANTIC_SCHEMA.items() if r.get("nullable")]
    assert nullable == ["total_bedrooms"]
    assert (
        SEMANTIC_SCHEMA["total_bedrooms"]["missing_treatment"]
        == "median_imputation_after_split"
    )


def test_validator_passes_on_clean_synthetic(synthetic_housing):
    issues, _ = validate_data(synthetic_housing)
    status, errors, _ = status_from_issues(issues)
    # Missing total_bedrooms is nullable -> at most warnings, never an error.
    assert errors == 0
    assert status in {"PASSED", "PASSED WITH WARNINGS"}


def test_validator_flags_out_of_range(synthetic_housing):
    bad = synthetic_housing.copy()
    bad.loc[bad.index[0], "latitude"] = 999.0  # above the 90.0 maximum
    issues, _ = validate_data(bad)
    status, errors, _ = status_from_issues(issues)
    assert errors >= 1
    assert status == "FAILED"


# ---------------------------------------------------------------------------
# The EDA stage is governed by this schema, and says so when it does not apply
# ---------------------------------------------------------------------------


def test_eda_refuses_a_dataset_the_schema_does_not_describe():
    """Unguarded, the schema-driven loops raised ``KeyError: 'ocean_proximity'`` and
    pandas' ``Cannot describe a DataFrame without columns`` from three frames deep, which
    reads as a broken stage. It is a dataset the governed schema does not cover, and the
    error has to say that."""
    import pandas as pd
    import pytest

    from src.pipelines.eda_pipeline import _require_schema_coverage

    churn_like = pd.DataFrame({"customerID": ["x"], "Contract": ["Month-to-month"]})
    with pytest.raises(ValueError, match="describes none of this dataset"):
        _require_schema_coverage(churn_like)


def test_eda_accepts_a_frame_the_schema_partly_describes(synthetic_housing):
    """Partial coverage is normal - engineered frames carry columns the schema never
    named - so only a total mismatch is refused."""
    from src.pipelines.eda_pipeline import _require_schema_coverage

    extended = synthetic_housing.copy()
    extended["rooms_per_household"] = 1.0
    _require_schema_coverage(extended)  # must not raise


def test_the_categorical_tables_skip_columns_the_frame_lacks(synthetic_housing):
    """The frequency loops index schema columns straight into the frame; on a project
    without them that was a KeyError rather than an empty table."""
    from src.eda.explorer import categorical_summary
    from src.eda.understanding import categorical_frequencies

    without = synthetic_housing.drop(columns=CATEGORICAL_COLUMNS)

    assert categorical_frequencies(without).empty
    assert categorical_summary(without).empty
    # And it still reports them when they are present.
    assert not categorical_frequencies(synthetic_housing).empty
