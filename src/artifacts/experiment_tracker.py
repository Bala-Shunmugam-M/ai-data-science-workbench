"""
src.artifacts.experiment_tracker
================================

PURPOSE
-------
Append-only experiment log: one JSON record per training run capturing the
model name, version, hyperparameters, metrics, wall-clock duration, and the
training-data content hash. Unlike the model registry (which holds the current
set of model versions), this file is an immutable history of every run.

PIPELINE POSITION
-----------------
Written by the training stage; read by reporting for the run history.

OUTPUTS
-------
    artifacts/experiments/experiments.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.paths import EXPERIMENTS_LOG_PATH, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_experiment(
    record: dict[str, Any],
    *,
    path: Path = EXPERIMENTS_LOG_PATH,
) -> dict[str, Any]:
    """Append one experiment record (stamped with a UTC time) and return it."""

    stamped = {"logged_at": utc_timestamp(), **record}
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stamped, ensure_ascii=False, default=str) + "\n")
    logger.info(
        "Logged experiment %s %s.",
        record.get("model"),
        record.get("version"),
    )
    return stamped


def read_experiments(path: Path = EXPERIMENTS_LOG_PATH) -> list[dict[str, Any]]:
    """Return every experiment record in order (empty when none exist)."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
