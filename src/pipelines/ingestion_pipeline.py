"""
src.pipelines.ingestion_pipeline
================================

PURPOSE
-------
Ingest the configured dataset (Kaggle or cached local file), structurally
profile it, and run semantic-schema validation.

PIPELINE POSITION
-----------------
Stage 1-3: Data Ingestion, (structural) Data Understanding, Validation.
"""

from __future__ import annotations

import pandas as pd

from config.paths import ensure_dirs
from src.data_manager.data_manager import ingest_dataset, validate_dataset
from src.domain.data_profile import write_profile
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(dataset_name: str = "california_housing", prefer_cache: bool = True) -> pd.DataFrame:
    """Ingest, profile, and validate the dataset; return the raw DataFrame."""

    ensure_dirs()
    logger.info("=== Ingestion pipeline: %s ===", dataset_name)

    dataset = ingest_dataset(dataset_name, prefer_cache=prefer_cache)
    logger.info("Ingested %s with shape %s", dataset_name, dataset.shape)

    write_profile(dataset.df, source_label=f"{dataset_name} (raw)")
    validate_dataset(dataset)
    return dataset.df


def run_validate(dataset_name: str = "california_housing") -> dict:
    """Load the cached raw dataset and (re)run validation as a standalone step."""

    ensure_dirs()
    dataset = ingest_dataset(dataset_name, prefer_cache=True)
    return validate_dataset(dataset)
