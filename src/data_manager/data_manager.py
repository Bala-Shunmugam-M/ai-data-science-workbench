"""
src.data_manager.data_manager
=============================

PURPOSE
-------
Facade over ingestion, validation, and raw access. Resolves a dataset name via
the registry, ingests it (Kaggle or cached local file), optionally validates
it, and returns a :class:`~src.domain.dataset.Dataset`.

PIPELINE POSITION
-----------------
Entry point used by the ingestion pipeline and by downstream stages that just
need "give me the validated raw frame".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import RAW_HOUSING_PATH
from src.connectors.file_connector import load_local_csv
from src.connectors.kaggle_connector import download_kaggle_dataset
from src.data_manager.data_validator import run_validation
from src.data_manager.dataset_registry import get_dataset
from src.domain.dataset import Dataset
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ingest_dataset(
    name: str = "california_housing",
    prefer_cache: bool = True,
    raw_path: Path = RAW_HOUSING_PATH,
) -> Dataset:
    """
    Ingest a registered dataset and return it as a :class:`Dataset`.

    When ``prefer_cache`` is True and ``raw_path`` already exists, the cached
    CSV is loaded (offline-friendly); otherwise the Kaggle connector downloads
    it. Falls back to the cache if the download fails.
    """

    descriptor = get_dataset(name)

    if prefer_cache and raw_path.exists():
        logger.info("Loading cached raw dataset from %s", raw_path)
        df = load_local_csv(raw_path)
        return Dataset(name=name, df=df)

    try:
        df = download_kaggle_dataset(descriptor.kaggle_id, descriptor.file, raw_path)
    except Exception as exc:  # noqa: BLE001 - fall back to cache when possible.
        if raw_path.exists():
            logger.warning("Kaggle download failed (%s); using cache %s", exc, raw_path)
            df = load_local_csv(raw_path)
        else:
            raise
    return Dataset(name=name, df=df)


def validate_dataset(dataset: Dataset, raise_on_failure: bool = False) -> dict:
    """Run semantic-schema validation on ``dataset.df`` and return the result."""

    return run_validation(dataset.df, raise_on_failure=raise_on_failure)


def get_raw_dataframe(raw_path: Path = RAW_HOUSING_PATH) -> pd.DataFrame:
    """Return the cached raw DataFrame, raising if ingestion has not run."""

    return load_local_csv(raw_path)
