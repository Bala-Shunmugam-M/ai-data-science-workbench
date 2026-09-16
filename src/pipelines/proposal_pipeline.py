"""
src.pipelines.proposal_pipeline
===============================

PURPOSE
-------
Stage 6: Model Proposal. Reads the engineered training split, builds the governed
model-proposal document (recommended candidates + validation strategy), seeds the
standing governance decisions, and audit-logs the event.

PIPELINE POSITION
-----------------
    feature engineering -> [model proposal] -> approval -> training

OUTPUTS
-------
    governance/approvals/model_proposal.json
"""

from __future__ import annotations

from typing import Any

from config.paths import ENGINEERED_TRAIN_PATH, ensure_dirs
from src.model_proposal.proposal import write_proposal
from src.utils.file_utils import load_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run() -> dict[str, Any]:
    """Write the model proposal from the engineered training split."""

    ensure_dirs()
    logger.info("=== Model proposal ===")
    train_df = load_csv(ENGINEERED_TRAIN_PATH)
    proposal = write_proposal(train_df)
    logger.info(
        "Model proposal complete: %d recommended model(s).",
        len(proposal["recommended_models"]),
    )
    return proposal
