"""Smoke tests for train-only median imputation (leakage prevention)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.imputer import MedianImputer
from src.preprocessing.splitter import split_data


def test_median_is_fit_on_train_only(synthetic_housing):
    train, validation, test = split_data(synthetic_housing)

    imputer = MedianImputer().fit(train)
    expected_median = float(
        pd.to_numeric(train["total_bedrooms"], errors="coerce").median()
    )
    assert imputer.medians_["total_bedrooms"] == expected_median

    # Force a missing value into validation and confirm it is filled with the
    # TRAIN-derived median (not validation's own median).
    validation = validation.copy()
    validation.loc[validation.index[0], "total_bedrooms"] = np.nan
    imputed = imputer.transform(validation)
    assert not imputed["total_bedrooms"].isna().any()
    assert imputed.loc[validation.index[0], "total_bedrooms"] == expected_median


def test_transform_before_fit_raises(synthetic_housing):
    imputer = MedianImputer()
    try:
        imputer.transform(synthetic_housing)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when transforming before fit.")


def test_fit_transform_removes_missing(synthetic_housing):
    train, _, _ = split_data(synthetic_housing)
    result = MedianImputer().fit_transform(train)
    assert not result["total_bedrooms"].isna().any()
