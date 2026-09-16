"""
src.eda.understanding
=====================

PURPOSE
-------
Document the structure and quality of the training set without modifying it:
a schema-aware variable summary (role, semantic_type, dtype, missing, unique,
example values, and numeric stats including skewness/kurtosis), a missing-value
summary, categorical frequencies, and logical-consistency checks.

PIPELINE POSITION
-----------------
    split -> [data understanding] -> EDA

Adapted from the professor's ``data_understanding.py``. Outputs are written to
``results/data_understanding/``; no observations or values are modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import RESULTS_DIR, ensure_dir
from src.domain.schema import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    TARGET_COLUMN,
)
from src.utils.file_utils import save_csv, save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

UNDERSTANDING_DIR: Path = RESULTS_DIR / "data_understanding"


def variable_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per-variable summary joining schema metadata with observed statistics."""

    rows: list[dict[str, object]] = []
    for column in data.columns:
        series = data[column]
        row: dict[str, object] = {
            "column_name": column,
            "role": SEMANTIC_SCHEMA.get(column, {}).get("role", ""),
            "semantic_type": SEMANTIC_SCHEMA.get(column, {}).get("semantic_type", ""),
            "pandas_dtype": str(series.dtype),
            "observations": len(data),
            "non_missing_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique(dropna=True)),
            "example_values": " | ".join(
                series.dropna().astype(str).drop_duplicates().head(5).tolist()
            ),
        }

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            row.update(
                {
                    "minimum": float(numeric.min()),
                    "maximum": float(numeric.max()),
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "standard_deviation": float(numeric.std(ddof=1)),
                    "skewness": float(numeric.skew()),
                    "kurtosis": float(numeric.kurt()),
                }
            )
        else:
            row.update(
                {
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "standard_deviation": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise columns with missing values and their schema treatment."""

    rows: list[dict[str, object]] = []
    for column in data.columns:
        count = int(data[column].isna().sum())
        if count == 0:
            continue
        rules = SEMANTIC_SCHEMA.get(column, {})
        rows.append(
            {
                "column_name": column,
                "missing_count": count,
                "missing_percentage": float(count / len(data) * 100),
                "nullable_in_schema": bool(rules.get("nullable", False)),
                "documented_treatment": rules.get("missing_treatment", ""),
            }
        )
    return pd.DataFrame(rows)


def categorical_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    """Frequency tables for each categorical column."""

    rows: list[dict[str, object]] = []
    # The semantic schema describes the housing dataset. On any other project the
    # unfiltered loop raised KeyError from inside the stage, which reads as a broken
    # pipeline rather than as "the schema does not describe this data".
    for column in [c for c in CATEGORICAL_COLUMNS if c in data.columns]:
        counts = (
            data[column].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        )
        for level, count in counts.items():
            rows.append(
                {
                    "column_name": column,
                    "category_level": str(level),
                    "frequency": int(count),
                    "percentage": float(count / len(data) * 100),
                }
            )
    return pd.DataFrame(rows)


def logical_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Domain and schema-bound consistency checks (PASS / REVIEW)."""

    rows: list[dict[str, object]] = []

    if {"households", "population"}.issubset(data.columns):
        count = int((data["households"] > data["population"]).sum())
        rows.append(
            {
                "check_name": "households_above_population",
                "affected_observations": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "description": "Household count should not normally exceed population.",
            }
        )

    if {"total_bedrooms", "total_rooms"}.issubset(data.columns):
        count = int(
            (data["total_bedrooms"].notna() & (data["total_bedrooms"] > data["total_rooms"])).sum()
        )
        rows.append(
            {
                "check_name": "bedrooms_above_rooms",
                "affected_observations": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "description": "Bedrooms should not normally exceed total rooms.",
            }
        )

    for column, rules in SEMANTIC_SCHEMA.items():
        if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column].dtype):
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")

        minimum = rules.get("minimum")
        if minimum is not None:
            count = int((numeric.notna() & (numeric < minimum)).sum())
            rows.append(
                {
                    "check_name": f"{column}_below_minimum",
                    "affected_observations": count,
                    "status": "PASS" if count == 0 else "REVIEW",
                    "description": f"Values below schema minimum {minimum}.",
                }
            )

        maximum = rules.get("maximum")
        if maximum is not None:
            count = int((numeric.notna() & (numeric > maximum)).sum())
            rows.append(
                {
                    "check_name": f"{column}_above_maximum",
                    "affected_observations": count,
                    "status": "PASS" if count == 0 else "REVIEW",
                    "description": f"Values above schema maximum {maximum}.",
                }
            )

        if rules.get("whole_number") is True:
            count = int((numeric.notna() & ~np.isclose(numeric, np.round(numeric))).sum())
            rows.append(
                {
                    "check_name": f"{column}_non_whole_values",
                    "affected_observations": count,
                    "status": "PASS" if count == 0 else "REVIEW",
                    "description": "Checks the whole-number requirement.",
                }
            )

    return pd.DataFrame(rows)


def run_understanding(
    train_df: pd.DataFrame,
    output_dir: Path = UNDERSTANDING_DIR,
) -> dict[str, pd.DataFrame]:
    """Compute all understanding tables, persist them, and return them."""

    ensure_dir(output_dir)

    summary = pd.DataFrame(
        [
            {
                "dataset": "Training",
                "observations": len(train_df),
                "variables": len(train_df.columns),
                "numerical_predictors": len(NUMERICAL_COLUMNS),
                "categorical_predictors": len(CATEGORICAL_COLUMNS),
                "target_column": TARGET_COLUMN,
                "total_missing_values": int(train_df.isna().sum().sum()),
                "variables_with_missing_values": int((train_df.isna().sum() > 0).sum()),
                "duplicate_rows": int(train_df.duplicated().sum()),
            }
        ]
    )

    variables = variable_summary(train_df)
    missing = missing_summary(train_df)
    categories = categorical_frequencies(train_df)
    checks = logical_checks(train_df)

    save_csv(summary, output_dir / "dataset_summary.csv")
    save_csv(variables, output_dir / "variable_summary.csv")
    save_csv(missing, output_dir / "missing_values.csv")
    save_csv(categories, output_dir / "categorical_frequencies.csv")
    save_csv(checks, output_dir / "logical_consistency_checks.csv")

    recommendations: list[str] = []
    if missing.empty:
        recommendations.append("No missing values were detected.")
    else:
        recommendations.append(
            "Missing values are present in: "
            + ", ".join(missing["column_name"].tolist())
            + ". Treat them only after the training split."
        )
    review_checks = checks.loc[checks["status"] == "REVIEW", "check_name"].tolist()
    if review_checks:
        recommendations.append("Logical checks requiring review: " + ", ".join(review_checks) + ".")
    else:
        recommendations.append("All implemented logical checks passed.")
    recommendations.append("No data were modified during data understanding.")

    text = (
        "DATA UNDERSTANDING SUMMARY\n"
        + "=" * 70 + "\n"
        + f"Observations: {len(train_df)}\n"
        + f"Variables: {len(train_df.columns)}\n"
        + f"Missing values: {int(train_df.isna().sum().sum())}\n"
        + f"Duplicate rows: {int(train_df.duplicated().sum())}\n"
        + "Recommendations:\n- " + "\n- ".join(recommendations) + "\n"
        + f"\nOutputs: {output_dir}\n"
        + "No observations or values were modified.\n"
        + "=" * 70 + "\n"
    )
    save_text(text, output_dir / "data_understanding_summary.txt")
    logger.info("Data understanding written to %s", output_dir)

    return {
        "summary": summary,
        "variables": variables,
        "missing": missing,
        "categories": categories,
        "checks": checks,
    }
