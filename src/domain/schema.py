"""
src.domain.schema
=================

PURPOSE
-------
Human-approved semantic schema for the housing dataset. This is the governed
contract that drives semantic validation, schema-driven preprocessing, data
understanding, and feature engineering.

PIPELINE POSITION
-----------------
Imported by the validator, preprocessing pipeline, data-understanding, EDA,
and feature-engineering stages. Ported verbatim from the professor's
``semantic_schema.py`` (clean module name in this codebase); the ``role``,
``semantic_type``, ``expected_storage_type``, ``nullable``, ``minimum``,
``maximum``, ``whole_number``, and ``preprocessing_group`` fields are preserved
exactly, including the ``median_imputation_after_split`` treatment for
``total_bedrooms``.
"""

from __future__ import annotations

SEMANTIC_SCHEMA = {
    "longitude": {
        "role": "predictor",
        "semantic_type": "spatial_coordinate",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": -180.0,
        "maximum": 180.0,
        "whole_number": False,
        "preprocessing_group": "numerical",
    },
    "latitude": {
        "role": "predictor",
        "semantic_type": "spatial_coordinate",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": -90.0,
        "maximum": 90.0,
        "whole_number": False,
        "preprocessing_group": "numerical",
    },
    "housing_median_age": {
        "role": "predictor",
        "semantic_type": "discrete_numeric",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": True,
        "preprocessing_group": "numerical",
    },
    "total_rooms": {
        "role": "predictor",
        "semantic_type": "count",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": True,
        "preprocessing_group": "numerical",
    },
    "total_bedrooms": {
        "role": "predictor",
        "semantic_type": "count",
        "expected_storage_type": "numeric",
        "nullable": True,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": True,
        "preprocessing_group": "numerical",
        # total_bedrooms may be missing; imputation is fit on train ONLY,
        # after the split, to avoid leakage.
        "missing_treatment": "median_imputation_after_split",
    },
    "population": {
        "role": "predictor",
        "semantic_type": "count",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": True,
        "preprocessing_group": "numerical",
    },
    "households": {
        "role": "predictor",
        "semantic_type": "count",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": True,
        "preprocessing_group": "numerical",
    },
    "median_income": {
        "role": "predictor",
        "semantic_type": "continuous_numeric",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": False,
        "preprocessing_group": "numerical",
    },
    "median_house_value": {
        "role": "target",
        "semantic_type": "continuous_numeric",
        "expected_storage_type": "numeric",
        "nullable": False,
        "minimum": 0.0,
        "maximum": None,
        "whole_number": False,
        "preprocessing_group": "target",
    },
    "ocean_proximity": {
        "role": "predictor",
        "semantic_type": "nominal_categorical",
        "expected_storage_type": "string",
        "nullable": False,
        "allowed_values": [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN",
        ],
        "preprocessing_group": "categorical",
    },
}


def schema_columns() -> list[str]:
    """Return the schema-approved columns in deterministic (schema) order."""

    return list(SEMANTIC_SCHEMA.keys())


def columns_by_group(group: str) -> list[str]:
    """Return columns whose ``preprocessing_group`` equals ``group``."""

    return [
        column
        for column, rules in SEMANTIC_SCHEMA.items()
        if rules.get("preprocessing_group") == group
    ]


TARGET_COLUMN: str = next(
    column
    for column, rules in SEMANTIC_SCHEMA.items()
    if rules.get("role") == "target"
)

NUMERICAL_COLUMNS: list[str] = columns_by_group("numerical")
CATEGORICAL_COLUMNS: list[str] = columns_by_group("categorical")

NON_NEGATIVE_COLUMNS: list[str] = [
    column
    for column, rules in SEMANTIC_SCHEMA.items()
    if rules.get("minimum") == 0.0
]

WHOLE_NUMBER_COLUMNS: list[str] = [
    column
    for column, rules in SEMANTIC_SCHEMA.items()
    if rules.get("whole_number") is True
]
