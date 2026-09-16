"""
src.connectors.file_connector
=============================

PURPOSE
-------
Load a dataset from a local CSV file. Enables fully offline runs (and testing)
when the raw data already exists on disk.

PIPELINE POSITION
-----------------
Alternative ingestion source to the Kaggle connector.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_local_csv(path: Path) -> pd.DataFrame:
    """Load a CSV from ``path``, raising a clear error when it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Local CSV not found: {path}")
    return pd.read_csv(path)
