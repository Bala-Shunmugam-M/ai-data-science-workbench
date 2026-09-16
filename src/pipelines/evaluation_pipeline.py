"""
src.pipelines.evaluation_pipeline
=================================

PURPOSE
-------
Stage 8: Model Evaluation. Compares all registered models on the validation
split, promotes the champion (best validation RMSE), and evaluates ONLY the
champion once on the untouched test split, writing the comparison table,
evaluation report, residual figures, and the final model-selection artifact.

PIPELINE POSITION
-----------------
    training -> [evaluation] -> explainability -> reporting

OUTPUTS
-------
    artifacts/evaluation/model_comparison.csv
    artifacts/evaluation/model_evaluation_report.json
    artifacts/evaluation/figures/*.png
    artifacts/final_model_selection.json
"""

from __future__ import annotations

from typing import Any

from config.paths import ensure_dirs
from src.model_evaluation.evaluator import evaluate
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(*, actor: str | None = None) -> dict[str, Any]:
    """Run the full evaluation stage and return the evaluation report."""

    ensure_dirs()
    logger.info("=== Model evaluation (champion on untouched test) ===")
    report = evaluate(actor=actor)
    champion = report.get("champion", {})
    logger.info(
        "Evaluation complete. Champion=%s %s.",
        champion.get("name"),
        champion.get("version"),
    )
    return report
