"""
src.utils.logging_utils
=======================

PURPOSE
-------
Provide a single ``get_logger`` factory that writes to both the console and a
rotating-free file log at ``artifacts/logs/workbench.log``.

PIPELINE POSITION
-----------------
Used by every pipeline stage so that a full run leaves a consolidated,
auditable trail on disk (supporting the governance/lineage requirement).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from config.paths import LOGS_DIR, ensure_dir

_LOG_FILE: Path = LOGS_DIR / "workbench.log"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger writing to console and the workbench log file.

    Handlers are attached once per logger name so repeated calls (common when
    stages are chained) do not duplicate log lines.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_name = os.environ.get("WORKBENCH_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ensure_dir(LOGS_DIR)
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
