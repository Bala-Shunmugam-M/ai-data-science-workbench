"""
src.reporting.report_generator
==============================

PURPOSE
-------
Gather every stage's on-disk artifacts into a single plain-Python context dict
that :mod:`src.reporting.html_reporter` renders to HTML. Keeping collection
(here) separate from rendering (there) makes the report data testable without a
browser and lets other renderers (e.g. PDF) reuse the same model later.

Every section degrades gracefully: a missing upstream artifact yields an empty
or "not available" section rather than an error, so a partial pipeline still
produces a readable report.

PIPELINE POSITION
-----------------
    explainability -> [reporting] -> deliverable
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import active
from config.constants import TARGET_COLUMN
from config.paths import (
    ARTIFACTS_DIR,
    EXECUTIVE_BRIEFING_PATH,
    FEATURE_APPROVAL_PATH,
    FEATURE_PROPOSAL_PATH,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_COMPARISON_CSV,
    MODEL_EVALUATION_REPORT_PATH,
    MODEL_PROPOSAL_PATH,
    RESULTS_DIR,
)
from src.explainability.interpretation import COEFFICIENT_CSV, TOP_DRIVERS_PNG
from src.governance import audit_logger, decision_tracker, lineage_tracker
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_csv, load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Key EDA figures worth embedding (subset of the full results/eda/figures set).
_KEY_EDA_FIGURES: list[tuple[str, str]] = [
    ("target_distribution.png", "Distribution of median house value (the target)."),
    (
        "median_income_vs_median_house_value.png",
        "Median income vs house value - the strongest single relationship.",
    ),
    ("correlation_matrix.png", "Correlation matrix across all numeric predictors."),
    (
        "geographic_house_value_distribution.png",
        "Geographic distribution of house value (coastal premium is visible).",
    ),
    (
        "median_house_value_by_ocean_proximity.png",
        "House value by ocean-proximity category.",
    ),
]


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (FileNotFoundError, ValueError):
        return None


def _safe_csv(path: Path) -> pd.DataFrame | None:
    try:
        return load_csv(path)
    except (FileNotFoundError, ValueError):
        return None


def _validation_section() -> dict[str, Any]:
    summary = _safe_csv(RESULTS_DIR / "validation" / "validation_summary.csv")
    status_file = RESULTS_DIR / "validation" / "validation_status.txt"
    status = status_file.read_text(encoding="utf-8").strip() if status_file.exists() else "n/a"
    return {
        "status": status,
        "summary": summary.to_dict(orient="records")[0] if summary is not None and not summary.empty else {},
    }


def _humanise(stem: str) -> str:
    """`median_income_distribution` -> `Median income distribution`."""

    text = stem.replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:] if text else stem


def _eda_section() -> list[dict[str, str]]:
    """Curated housing figures when present, otherwise whatever the project has.

    The curated list carries hand-written captions for the graded project. An
    uploaded dataset has none of those filenames, and returning an empty list
    left its report with no figures at all - so fall back to the auto-EDA
    output the generic pipeline writes, captioned from the file name.
    """

    figures: list[dict[str, str]] = []
    figures_dir = RESULTS_DIR / "eda" / "figures"
    for filename, caption in _KEY_EDA_FIGURES:
        path = figures_dir / filename
        if path.exists():
            figures.append({"path": str(path), "caption": caption})
    if figures:
        return figures

    for directory in (ARTIFACTS_DIR / "auto_eda", figures_dir):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.png")):
            figures.append({"path": str(path), "caption": _humanise(path.stem)})
    return figures


def _feature_section() -> dict[str, Any]:
    proposal = _safe_json(FEATURE_PROPOSAL_PATH) or {}
    approval = _safe_json(FEATURE_APPROVAL_PATH) or {}
    features = proposal.get("features", [])
    plan_rows = [
        {
            "feature_name": f.get("feature_name"),
            "formula": f.get("formula"),
            "business_meaning": f.get("business_meaning"),
            "automatic": f.get("automatic"),
            "approved": f.get("feature_name") in set(approval.get("approved_features", [])),
        }
        for f in features
    ]
    return {
        "total_proposed": proposal.get("total_proposed", len(features)),
        "approved_features": approval.get("approved_features", []),
        "approved_predictor_columns": approval.get("approved_predictor_columns", []),
        "plan": plan_rows,
    }


def _model_section() -> dict[str, Any]:
    proposal = _safe_json(MODEL_PROPOSAL_PATH) or {}
    approval = _safe_json(MODEL_APPROVAL_PATH) or {}
    comparison = _safe_csv(MODEL_COMPARISON_CSV)
    evaluation = _safe_json(MODEL_EVALUATION_REPORT_PATH) or {}
    final = _safe_json(FINAL_MODEL_SELECTION_PATH) or {}
    return {
        "recommended_models": proposal.get("recommended_models", []),
        "approved_models": approval.get("approved_models", []),
        "validation_strategy": proposal.get("validation_strategy", {}),
        "comparison": comparison.to_dict(orient="records") if comparison is not None else [],
        "champion": evaluation.get("champion", {}),
        "selection_narrative": evaluation.get("selection_narrative", ""),
        "residual_figures": evaluation.get("residual_figures", {}),
        "final_selection": final,
    }


def _coefficient_section() -> dict[str, Any]:
    table = _safe_csv(COEFFICIENT_CSV)
    rows = table.to_dict(orient="records") if table is not None else []
    return {
        "coefficients": rows,
        "top_drivers_png": str(TOP_DRIVERS_PNG) if TOP_DRIVERS_PNG.exists() else None,
    }


def _governance_section() -> dict[str, Any]:
    return {
        "audit_summary": audit_logger.summarize_events(),
        "decisions": decision_tracker.list_decisions(),
        "lineage": lineage_tracker.summarize(),
        "lineage_nodes": lineage_tracker.list_nodes(),
    }


def _briefing_section() -> str:
    if EXECUTIVE_BRIEFING_PATH.exists():
        return EXECUTIVE_BRIEFING_PATH.read_text(encoding="utf-8")
    return ""


def build_report_context() -> dict[str, Any]:
    """Collect all stage artifacts into a single render-ready context dict."""

    logger.info("Collecting report context from stage artifacts.")

    # Describe the project that is actually active. Hard-coding the housing
    # title meant every uploaded dataset would have produced a report headed
    # "California Housing Project Report" about a different target entirely.
    try:
        descriptor = active.active_dataset()
        display_name = descriptor.display_name or descriptor.name
        target = active.target_column()
        task_name = active.task()
    except Exception:  # noqa: BLE001 - no descriptor: fall back to the showcase.
        display_name = "California Housing"
        target = TARGET_COLUMN
        task_name = "regression"

    scope = (
        "Linear family (LinearRegression, Ridge, Lasso); catalogue extensible."
        if task_name == "regression"
        else "Classification catalogue (LogisticRegression, DecisionTree, RandomForest)."
    )

    return {
        "generated_at": utc_timestamp(),
        "target_column": target,
        "overview": {
            "title": f"AI Data Science Workbench - {display_name} Project Report",
            "subtitle": f"Governed, reproducible {task_name} on {target}",
            "scope": scope,
        },
        "validation": _validation_section(),
        "eda_figures": _eda_section(),
        "features": _feature_section(),
        "models": _model_section(),
        "coefficients": _coefficient_section(),
        "governance": _governance_section(),
        "executive_briefing_md": _briefing_section(),
    }
