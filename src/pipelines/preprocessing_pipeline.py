"""
src.pipelines.preprocessing_pipeline
====================================

PURPOSE
-------
Deterministic cleaning -> stratified split -> train-only fitted transforms.

PIPELINE POSITION
-----------------
Stage 4: Preprocessing. Produces ``data/processed/housing_clean.csv`` and the
``data/splits/{train,validation,test}.csv`` files, then fits the imputer,
encoder, and scaler on the training split only.
"""

from __future__ import annotations

from config.paths import (
    PROCESSED_HOUSING_PATH,
    RAW_HOUSING_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    ensure_dirs,
)
from src.preprocessing.pipeline import (
    FittedTransformers,
    clean_dataset,
    save_clean_dataset,
)
from src.preprocessing.splitter import save_splits, split_data, validate_splits
from src.utils.file_utils import load_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_clean() -> None:
    """Clean the raw dataset into the processed schema-aligned dataset."""

    ensure_dirs()
    logger.info("=== Preprocessing: cleaning ===")
    raw = load_csv(RAW_HOUSING_PATH)
    clean, unexpected, changes = clean_dataset(raw)
    if unexpected:
        logger.info("Excluded unexpected columns: %s", unexpected)
    save_clean_dataset(clean)


def run_split() -> None:
    """Split the cleaned dataset into stratified train/validation/test files."""

    ensure_dirs()
    logger.info("=== Preprocessing: split ===")
    clean = load_csv(PROCESSED_HOUSING_PATH)
    train_df, validation_df, test_df = split_data(clean)
    validate_splits(clean, train_df, validation_df, test_df)
    save_splits(train_df, validation_df, test_df)


def run_fit_transforms() -> FittedTransformers:
    """Fit imputer/encoder/scaler on the training split only."""

    logger.info("=== Preprocessing: fit train-only transforms ===")
    train_df = load_csv(TRAIN_PATH)
    transformers = FittedTransformers().fit(train_df)
    return transformers


def run() -> None:
    """Full preprocessing stage: clean -> split -> fit transforms."""

    run_clean()
    run_split()
    run_fit_transforms()
    logger.info("Preprocessing complete. Splits at %s", TRAIN_PATH.parent)
