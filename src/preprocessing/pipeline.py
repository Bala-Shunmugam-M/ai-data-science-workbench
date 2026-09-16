"""
src.preprocessing.pipeline
==========================

PURPOSE
-------
Compose the preprocessing stage: deterministic, schema-driven cleaning that
produces ``data/processed/housing_clean.csv``, plus a composition helper that
fits the train-only transformations (imputer -> encoder -> scaler).

PIPELINE POSITION
-----------------
    validation -> [clean] -> split -> [fit transforms on train]

Adapted from the professor's ``preprocess.py``. The cleaning is schema-driven
(no hard-coded column lists): retain and order schema columns, coerce numerics,
standardise categorical text, and enforce the schema's nullability, bounds, and
whole-number policies. Allowed missing values (``total_bedrooms``) are
PRESERVED here on purpose; imputation is fit later on train only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.paths import PROCESSED_HOUSING_PATH, ensure_dir
from src.domain.schema import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SEMANTIC_SCHEMA,
    schema_columns,
)
from src.preprocessing.encoder import OneHotCategoricalEncoder
from src.preprocessing.imputer import MedianImputer
from src.preprocessing.scaler import NumericStandardScaler
from src.utils.file_utils import save_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deterministic, schema-driven cleaning (adapted from preprocess.py).
# ---------------------------------------------------------------------------

def _require_schema_columns(data: pd.DataFrame) -> None:
    missing = [c for c in schema_columns() if c not in data.columns]
    if missing:
        raise KeyError("Missing schema columns: " + ", ".join(missing))


def _select_and_order(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    approved = schema_columns()
    unexpected = [c for c in data.columns if c not in approved]
    return data.loc[:, approved].copy(), unexpected


def _coerce_numeric(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.copy()
    target_columns = [
        c for c, r in SEMANTIC_SCHEMA.items() if r.get("preprocessing_group") == "target"
    ]
    numeric_columns = list(dict.fromkeys(NUMERICAL_COLUMNS + target_columns))

    for column in numeric_columns:
        before = int(clean[column].isna().sum())
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        after = int(clean[column].isna().sum())
        if after - before > 0:
            raise ValueError(
                f"Numeric coercion introduced {after - before} missing value(s) "
                f"in '{column}', indicating invalid numeric content."
            )
    return clean


def _standardise_categorical(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    clean = data.copy()
    changed_counts: dict[str, int] = {}

    for column in CATEGORICAL_COLUMNS:
        original = clean[column].copy()
        missing_mask = original.isna()
        cleaned = original.astype("string").str.strip().mask(missing_mask, pd.NA)

        empty_mask = cleaned.notna() & cleaned.eq("")
        if int(empty_mask.sum()) > 0:
            raise ValueError(
                f"Categorical column '{column}' has {int(empty_mask.sum())} "
                "empty value(s) after trimming."
            )

        changed = (
            original.astype("string").fillna("<MISSING>")
            .ne(cleaned.astype("string").fillna("<MISSING>"))
        )
        changed_counts[column] = int(changed.sum())
        clean[column] = cleaned

    return clean, changed_counts


def _enforce_policies(data: pd.DataFrame) -> None:
    """Defensive schema checks (nullability, bounds, whole-number)."""

    for column, rules in SEMANTIC_SCHEMA.items():
        series = data[column]
        if not rules.get("nullable", False) and int(series.isna().sum()) > 0:
            raise ValueError(f"Non-nullable column '{column}' contains missing values.")

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            minimum, maximum = rules.get("minimum"), rules.get("maximum")
            if minimum is not None and int((numeric.notna() & (numeric < minimum)).sum()):
                raise ValueError(f"'{column}' has values below minimum {minimum}.")
            if maximum is not None and int((numeric.notna() & (numeric > maximum)).sum()):
                raise ValueError(f"'{column}' has values above maximum {maximum}.")
            if rules.get("whole_number") is True:
                mask = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
                if int(mask.sum()):
                    raise ValueError(f"'{column}' violates the whole-number rule.")


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """
    Produce the cleaned, schema-aligned frame from the raw dataset.

    Returns ``(clean_df, unexpected_columns, categorical_change_counts)``.
    No rows are removed and allowed missing values are preserved.
    """

    _require_schema_columns(raw_df)
    clean, unexpected = _select_and_order(raw_df)
    clean = _coerce_numeric(clean)
    clean, categorical_changes = _standardise_categorical(clean)
    _enforce_policies(clean)

    if len(clean) != len(raw_df):
        raise RuntimeError("Cleaning unexpectedly changed the number of rows.")

    return clean, unexpected, categorical_changes


def save_clean_dataset(clean_df: pd.DataFrame) -> None:
    """Write the cleaned dataset to ``data/processed/housing_clean.csv``."""

    ensure_dir(PROCESSED_HOUSING_PATH.parent)
    save_csv(clean_df, PROCESSED_HOUSING_PATH)
    logger.info("Saved cleaned dataset to %s", PROCESSED_HOUSING_PATH)


# ---------------------------------------------------------------------------
# Train-only transformation composition (imputer -> encoder -> scaler).
# ---------------------------------------------------------------------------

class FittedTransformers:
    """Bundle of the three train-fitted transformers, applied in order."""

    def __init__(self) -> None:
        self.imputer = MedianImputer()
        self.encoder = OneHotCategoricalEncoder()
        self.scaler = NumericStandardScaler()

    def fit(self, train_df: pd.DataFrame) -> "FittedTransformers":
        """Fit every transformer on the training split only."""

        imputed = self.imputer.fit(train_df).transform(train_df)
        self.encoder.fit(imputed)
        # Scaler is fit on the imputed (pre-encoding) numeric columns.
        self.scaler.fit(imputed)
        self.imputer.save()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputer -> scaler -> encoder to any split with the fitted stats."""

        result = self.imputer.transform(df)
        result = self.scaler.transform(result)
        result = self.encoder.transform(result)
        return result
