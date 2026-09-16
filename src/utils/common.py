"""
src.utils.common
================

PURPOSE
-------
Generic, domain-neutral utilities: timestamps and lightweight YAML loading.

PIPELINE POSITION
-----------------
Used by the dataset registry (YAML) and by stages that stamp reports/approvals
with a generation time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp (seconds precision)."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict, raising a clear error when missing."""

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
