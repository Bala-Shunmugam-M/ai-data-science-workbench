"""
src.feature_engineering.transformers
=====================================

PURPOSE
-------
Deterministic feature constructors used to execute the approved
(``automatic=True``) proposals: ratio features and log transforms. These are
pure functions of a single row's values, so they are leakage-free and require
no fitting.

PIPELINE POSITION
-----------------
    feature proposal -> [transformers] (invoked by the feature pipeline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.data_utils import safe_divide

# Ratio feature name -> (numerator, denominator).
RATIO_FEATURES: dict[str, tuple[str, str]] = {
    "rooms_per_household": ("total_rooms", "households"),
    "bedrooms_per_room": ("total_bedrooms", "total_rooms"),
    "population_per_household": ("population", "households"),
    "bedrooms_per_household": ("total_bedrooms", "households"),
    "rooms_per_person": ("total_rooms", "population"),
    "bedrooms_per_person": ("total_bedrooms", "population"),
}

# Log feature name -> source column.
LOG_FEATURES: dict[str, str] = {
    "log_total_rooms": "total_rooms",
    "log_total_bedrooms": "total_bedrooms",
    "log_population": "population",
    "log_households": "households",
}


def add_ratio_feature(df: pd.DataFrame, name: str) -> pd.Series:
    """Compute one ratio feature, returning NaN where the denominator is zero."""

    numerator, denominator = RATIO_FEATURES[name]
    return safe_divide(df[numerator], df[denominator])


def add_log_feature(df: pd.DataFrame, name: str) -> pd.Series:
    """Compute one log1p feature for a non-negative count column."""

    source = LOG_FEATURES[name]
    return np.log1p(pd.to_numeric(df[source], errors="coerce"))


def apply_automatic_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a copy of ``df`` with each requested automatic feature appended.

    Unknown feature names are ignored (the pipeline only passes approved
    deterministic ratios and log transforms).
    """

    result = df.copy()
    for name in feature_names:
        if name in RATIO_FEATURES:
            result[name] = add_ratio_feature(df, name)
        elif name in LOG_FEATURES:
            result[name] = add_log_feature(df, name)
    return result
