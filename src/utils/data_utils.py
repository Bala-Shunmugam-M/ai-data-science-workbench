"""
src.utils.data_utils
====================

PURPOSE
-------
Small, reusable dataframe helpers shared by the data-understanding, EDA, and
feature-engineering stages (numeric coercion, safe division, percentage,
modified Z-scores). These reproduce calculations the professor implemented
inline so the logic exists in exactly one place.

PIPELINE POSITION
-----------------
Used by domain profiling, EDA, and feature transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pct(numerator: float, denominator: float) -> float:
    """Percentage helper returning 0.0 when the denominator is zero."""

    return 0.0 if denominator == 0 else numerator / denominator * 100.0


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of numeric-dtype columns in ``df``."""

    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column].dtype)
    ]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Element-wise division that yields NaN where the denominator is zero.

    Used by deterministic ratio features so a zero denominator never produces
    an infinite value that would corrupt downstream scaling.
    """

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def modified_z_scores(series: pd.Series) -> pd.Series:
    """
    Modified Z-scores using the median absolute deviation (MAD).

    Returns all-NaN when the MAD is zero or undefined. Reproduces the
    professor's ``calculate_modified_z_scores`` (constant 0.6745).
    """

    numeric = pd.to_numeric(series, errors="coerce")
    median_value = numeric.median()
    mad = (numeric - median_value).abs().median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(float("nan"), index=series.index, dtype="float64")

    return 0.6745 * (numeric - median_value) / mad
