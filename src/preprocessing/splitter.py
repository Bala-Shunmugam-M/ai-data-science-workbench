"""
src.preprocessing.splitter
==========================

PURPOSE
-------
Split the cleaned dataset into train / validation / test using a stratified
scheme on binned ``median_income``.

PIPELINE POSITION
-----------------
    preprocessing (clean) -> [split] -> train-only transforms

Adapted from the professor's ``split.py``. The exact behaviour is preserved:

- a temporary ``income_category`` is created with ``pd.cut`` on
  ``median_income`` using bins ``[0, 1.5, 3.0, 4.5, 6.0, inf]``;
- a 70 / 15 / 15 stratified split via two ``train_test_split`` calls with
  ``random_state=42`` (0.30 then 0.50);
- the temporary ``income_category`` column is dropped from the saved splits;
- splits are written to ``data/splits/{train,validation,test}.csv``.

Bins, labels, seed, and fractions come from :mod:`config.constants`.
"""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import (
    INCOME_BINS,
    INCOME_LABELS,
    RANDOM_STATE,
    TEMP_TEST_SIZE,
    VALIDATION_TEST_SPLIT,
)
from config.paths import TEST_PATH, TRAIN_PATH, VALIDATION_PATH, ensure_dir
from src.utils.file_utils import save_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

INCOME_CATEGORY_COLUMN = "income_category"


def add_income_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the temporary ``income_category`` used only for stratification.

    The category preserves the median-income distribution across the three
    splits and is dropped before the splits are saved.
    """

    stratified_df = df.copy()
    stratified_df[INCOME_CATEGORY_COLUMN] = pd.cut(
        stratified_df["median_income"],
        bins=INCOME_BINS,
        labels=INCOME_LABELS,
        include_lowest=True,
    )

    if stratified_df[INCOME_CATEGORY_COLUMN].isna().any():
        raise ValueError(
            "Some observations could not be assigned to an income category."
        )
    return stratified_df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into 70% train, 15% validation, 15% test, stratified on income.

    The temporary ``income_category`` column is removed from every returned
    split.
    """

    stratified_df = add_income_category(df)

    train_df, temporary_df = train_test_split(
        stratified_df,
        test_size=TEMP_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratified_df[INCOME_CATEGORY_COLUMN],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=VALIDATION_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=temporary_df[INCOME_CATEGORY_COLUMN],
    )

    def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=INCOME_CATEGORY_COLUMN).reset_index(drop=True)

    return _finalise(train_df), _finalise(validation_df), _finalise(test_df)


#: Derived, not guessed. The binding constraint is the *second* split: sklearn
#: needs >= 2 rows of every class in the frame it is given, and that frame is
#: the temporary portion holding TEMP_TEST_SIZE of each class. So a class of
#: size c must satisfy ``c * TEMP_TEST_SIZE >= 2``.
#:
#: At the current 0.30 that is 7 rows per class. An earlier version of this
#: guard hard-coded 4, which passed the first split and then let sklearn raise
#: its own error on the second - the exact failure this exists to prevent.
_MIN_ROWS_PER_CLASS = math.ceil(2 / TEMP_TEST_SIZE)


def _require_stratifiable(labels: pd.Series, target_column: str) -> None:
    """Fail with an actionable message before sklearn raises a cryptic one.

    sklearn's own error ("The least populated class in y has only 1 member")
    names neither the column nor the offending classes, and surfaces in the GUI
    as a bare traceback. A user who uploaded a CSV needs to know which target
    they picked, which values are too rare, and what to do about it.
    """

    counts = labels.value_counts(dropna=False)
    too_rare = counts[counts < _MIN_ROWS_PER_CLASS]
    if too_rare.empty:
        return

    listed = ", ".join(f"{value!r} ({count})" for value, count in too_rare.items())
    raise ValueError(
        f"Cannot split on '{target_column}': "
        f"{len(too_rare)} of {len(counts)} classes have fewer than "
        f"{_MIN_ROWS_PER_CLASS} rows - {listed}. A stratified 70/15/15 split "
        f"needs at least {_MIN_ROWS_PER_CLASS} rows per class so every class "
        "reaches train, validation and test. Either drop those rows, merge the "
        "rare classes into an 'other' category, or choose a target with fewer "
        "distinct values."
    )


def split_by_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    70/15/15 split stratified directly on ``target_column``.

    Used by classification projects (e.g. churn) so the class balance is
    preserved across train/validation/test. Same seed and fractions as the
    income-stratified housing split; no temporary column is added.
    """

    _require_stratifiable(df[target_column], target_column)

    train_df, temporary_df = train_test_split(
        df,
        test_size=TEMP_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target_column],
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=VALIDATION_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=temporary_df[target_column],
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_random(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Plain (unstratified) 70/15/15 split — used by regression on an arbitrary
    uploaded dataset where no domain stratification variable is known.
    """

    train_df, temporary_df = train_test_split(
        df, test_size=TEMP_TEST_SIZE, random_state=RANDOM_STATE
    )
    validation_df, test_df = train_test_split(
        temporary_df, test_size=VALIDATION_TEST_SPLIT, random_state=RANDOM_STATE
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_splits(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Confirm the splits partition the rows and preserve the column set."""

    total = len(train_df) + len(validation_df) + len(test_df)
    if total != len(original_df):
        raise ValueError(
            "The total number of rows in the splits does not match the original."
        )

    expected_columns = set(original_df.columns)
    for split_name, split_df in {
        "training": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        if set(split_df.columns) != expected_columns:
            raise ValueError(f"The {split_name} set has incorrect columns.")


def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist the three splits to ``data/splits/``."""

    ensure_dir(TRAIN_PATH.parent)
    save_csv(train_df, TRAIN_PATH)
    save_csv(validation_df, VALIDATION_PATH)
    save_csv(test_df, TEST_PATH)
    logger.info(
        "Saved splits: train=%d, validation=%d, test=%d",
        len(train_df),
        len(validation_df),
        len(test_df),
    )
