"""
src.pipelines.reporting_pipeline
================================

PURPOSE
-------
Explainability + reporting. Interprets the champion linear model (standardized
coefficients, top-drivers chart, executive briefing), optionally runs SHAP if it
is installed, and assembles the single self-contained HTML project report.

PIPELINE POSITION
-----------------
    evaluation -> [explainability + reporting] -> deliverable

OUTPUTS
-------
    artifacts/reports/executive_briefing.md
    artifacts/explainability/*
    artifacts/reports/project_report.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.paths import ensure_dirs
from src.explainability import interpretation, shap_explainer
from src.reporting import html_reporter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_explain(*, actor: str | None = None) -> dict[str, Any]:
    """Interpret the champion model and (optionally) run SHAP."""

    ensure_dirs()
    logger.info("=== Explainability (champion interpretation) ===")
    summary = interpretation.interpret_champion(actor=actor)
    # Optional, dependency-guarded; a no-op when shap is not installed.
    summary["shap"] = shap_explainer.explain_champion_with_shap()
    return summary


def run_report() -> Path:
    """Assemble the single self-contained HTML project report."""

    ensure_dirs()
    logger.info("=== Reporting (self-contained HTML) ===")
    return html_reporter.write_report()


def run(*, actor: str | None = None) -> Path:
    """Full reporting stage: explain the champion, then write the HTML report."""

    run_explain(actor=actor)
    return run_report()
