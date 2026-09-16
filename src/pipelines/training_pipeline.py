"""
src.pipelines.training_pipeline
===============================

PURPOSE
-------
Stage 7: Model Training. Trains every APPROVED model on the engineered training
split, tuning Ridge/Lasso alpha on the validation split, and versions/registers
each fitted model. Refuses to run for any unapproved model (``GovernanceError``).

PIPELINE POSITION
-----------------
    approval -> [training] -> evaluation

OUTPUTS
-------
    models/<name>/<version>/{model.joblib,metadata.json}
    models/model_registry.json
    artifacts/experiments/experiments.jsonl
"""

from __future__ import annotations

from typing import Any

from config.paths import ensure_dirs
from src.model_training.trainer import train_all
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(*, actor: str | None = None) -> list[dict[str, Any]]:
    """Train every approved model and return their registry entries."""

    ensure_dirs()
    logger.info("=== Model training (approved models only) ===")
    entries = train_all(actor=actor)
    logger.info("Training complete: %d model(s) trained.", len(entries))
    return entries
