"""Smoke tests for the stratified splitter logic."""

from __future__ import annotations

import pandas as pd

from config.constants import INCOME_BINS
from src.preprocessing.splitter import (
    INCOME_CATEGORY_COLUMN,
    add_income_category,
    split_data,
)


def test_income_category_uses_config_bins(synthetic_housing):
    categorised = add_income_category(synthetic_housing)
    assert INCOME_CATEGORY_COLUMN in categorised.columns
    assert not categorised[INCOME_CATEGORY_COLUMN].isna().any()
    # Five bins -> five labels present given the spread-out synthetic income.
    assert categorised[INCOME_CATEGORY_COLUMN].nunique() == len(INCOME_BINS) - 1


def test_split_partitions_rows_and_drops_helper(synthetic_housing):
    train, validation, test = split_data(synthetic_housing)

    # Partition: rows conserved, no overlap in the helper column.
    assert len(train) + len(validation) + len(test) == len(synthetic_housing)
    for split in (train, validation, test):
        assert INCOME_CATEGORY_COLUMN not in split.columns
        assert set(split.columns) == set(synthetic_housing.columns)

    # Approximate 70/15/15 proportions.
    total = len(synthetic_housing)
    assert abs(len(train) / total - 0.70) < 0.02
    assert abs(len(validation) / total - 0.15) < 0.02
    assert abs(len(test) / total - 0.15) < 0.02


def test_split_is_deterministic(synthetic_housing):
    first = split_data(synthetic_housing)[0].reset_index(drop=True)
    second = split_data(synthetic_housing)[0].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_stratification_preserves_income_distribution(synthetic_housing):
    categorised = add_income_category(synthetic_housing)
    overall = categorised[INCOME_CATEGORY_COLUMN].value_counts(normalize=True).sort_index()

    train, _, _ = split_data(synthetic_housing)
    train_cat = add_income_category(train)[INCOME_CATEGORY_COLUMN]
    train_dist = train_cat.value_counts(normalize=True).sort_index()

    # Stratified split keeps each bin's share close to the population share.
    for label in overall.index:
        assert abs(train_dist.get(label, 0.0) - overall[label]) < 0.05
