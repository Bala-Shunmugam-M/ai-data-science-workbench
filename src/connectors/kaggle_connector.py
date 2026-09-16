"""
src.connectors.kaggle_connector
===============================

PURPOSE
-------
Download the California Housing dataset from Kaggle via ``kagglehub`` and copy
the raw CSV into ``data/raw/housing.csv``.

PIPELINE POSITION
-----------------
First step of the ingestion pipeline. Adapted from the professor's
``ingest.py``: the ``kagglehub.dataset_download("harrywang/housing")`` call,
the copy to ``data/raw/housing.csv``, and returning the loaded DataFrame are
preserved. The download id and file name are read from ``datasets.yaml`` via
the dataset registry rather than hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import DATA_RAW, RAW_HOUSING_PATH, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def download_kaggle_dataset(
    kaggle_id: str,
    file_name: str,
    destination: Path = RAW_HOUSING_PATH,
) -> pd.DataFrame:
    """
    Download ``file_name`` from the Kaggle dataset ``kaggle_id`` and cache it.

    The file is copied to ``destination`` (default ``data/raw/housing.csv``)
    and returned as a DataFrame. ``kagglehub`` is imported lazily so the rest
    of the platform works offline when the download is not needed.
    """

    import kagglehub  # lazy import: only required when actually downloading.

    ensure_dir(DATA_RAW)

    logger.info("Downloading Kaggle dataset '%s' ...", kaggle_id)
    dataset_dir = Path(kagglehub.dataset_download(kaggle_id))
    source_file = dataset_dir / file_name

    if not source_file.exists():
        raise FileNotFoundError(
            f"Expected '{file_name}' in downloaded dataset directory "
            f"{dataset_dir}, but it was not found."
        )

    df = pd.read_csv(source_file)
    ensure_dir(destination.parent)
    df.to_csv(destination, index=False)

    logger.info("Raw dataset cached to %s (shape=%s)", destination, df.shape)
    return df
