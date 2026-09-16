"""
src.pipelines.feature_pipeline
==============================

PURPOSE
-------
Governed feature engineering: write the feature proposal, then execute only the
approved (``automatic=True``) features into the engineered datasets.

PIPELINE POSITION
-----------------
Stage 5: Feature Engineering. Reads the splits, writes the proposal to
governance + results, and writes ``data/engineered/{train,validation,test}.csv``.
"""

from __future__ import annotations

from config.paths import TEST_PATH, TRAIN_PATH, VALIDATION_PATH, ensure_dirs
from src.feature_engineering.feature_proposal import write_proposal
from src.feature_engineering.pipeline import execute_features
from src.utils.file_utils import load_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_proposal() -> list[dict]:
    """Write the feature proposal from the training split."""

    ensure_dirs()
    logger.info("=== Feature proposal ===")
    train_df = load_csv(TRAIN_PATH)
    return write_proposal(train_df)


def run() -> None:
    """Full feature stage: propose, then execute approved features."""

    ensure_dirs()
    proposal = run_proposal()

    logger.info("=== Feature execution (automatic only) ===")
    train_df = load_csv(TRAIN_PATH)
    validation_df = load_csv(VALIDATION_PATH)
    test_df = load_csv(TEST_PATH)
    execute_features(proposal, train_df, validation_df, test_df)
    logger.info("Feature engineering complete.")
