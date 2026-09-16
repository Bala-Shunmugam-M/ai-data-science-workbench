"""
src.utils.file_utils
====================

PURPOSE
-------
Filesystem I/O helpers that auto-create parent directories and return the
written path. Centralising this keeps every stage's save/load behaviour
consistent (UTF-8, ``index=False`` for tabular data) and reproducible.

PIPELINE POSITION
-----------------
Used by all stages that persist datasets, reports, or fitted-artifact metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` if needed and return ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write ``df`` to ``path`` as UTF-8 CSV, creating parents. Returns path."""

    ensure_parent(path)
    df.to_csv(path, index=index, encoding="utf-8")
    return path


def save_json(data: Any, path: Path, indent: int = 2) -> Path:
    """Write ``data`` to ``path`` as pretty UTF-8 JSON, creating parents."""

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, default=str)
    return path


def save_text(text: str, path: Path) -> Path:
    """Write ``text`` to ``path`` as UTF-8, creating parents. Returns path."""

    ensure_parent(path)
    path.write_text(text, encoding="utf-8")
    return path


def load_csv(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV, raising a clear error when the file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def load_json(path: Path) -> Any:
    """Load JSON, raising a clear error when the file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
