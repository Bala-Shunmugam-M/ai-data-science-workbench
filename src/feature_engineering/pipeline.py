"""
src.feature_engineering.pipeline
================================

PURPOSE
-------
Execute ONLY the approved (``automatic=True``) features from the feature
proposal and write the engineered train/validation/test datasets.

PIPELINE POSITION
-----------------
    feature proposal -> [feature execution] -> engineered datasets

LEAKAGE PREVENTION
------------------
The automatic features here (deterministic ratios and log transforms) are pure
row-wise functions and need no fitting, so applying them to each split
independently cannot leak information. Any future feature requiring a fitted
statistic must be fit on the training split only (the proposal records this via
``requires_training_fit``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import (
    ENGINEERED_TEST_PATH,
    ENGINEERED_TRAIN_PATH,
    ENGINEERED_VALIDATION_PATH,
    ensure_dir,
)
from src.feature_engineering.transformers import apply_automatic_features
from src.utils.file_utils import save_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def automatic_feature_names(proposal: list[dict[str, Any]]) -> list[str]:
    """Return the names of features flagged automatic in the proposal."""

    return [row["feature_name"] for row in proposal if row.get("automatic")]


def execute_features(
    proposal: list[dict[str, Any]],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Apply the approved automatic features to all three splits."""

    names = automatic_feature_names(proposal)
    logger.info("Executing %d automatic feature(s): %s", len(names), names)

    engineered = {
        "train": apply_automatic_features(train_df, names),
        "validation": apply_automatic_features(validation_df, names),
        "test": apply_automatic_features(test_df, names),
    }

    ensure_dir(ENGINEERED_TRAIN_PATH.parent)
    save_csv(engineered["train"], ENGINEERED_TRAIN_PATH)
    save_csv(engineered["validation"], ENGINEERED_VALIDATION_PATH)
    save_csv(engineered["test"], ENGINEERED_TEST_PATH)
    logger.info(
        "Saved engineered splits to %s", ENGINEERED_TRAIN_PATH.parent
    )
    return engineered
