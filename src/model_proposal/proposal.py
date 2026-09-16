"""
src.model_proposal.proposal
===========================

PURPOSE
-------
Stage 6: build the Model Proposal document. It inspects the engineered training
data (row count, predictor count, and the multicollinearity flags produced by
the EDA analytical summary) and produces a governed proposal that lists the
recommended candidate models, the rationale for each, and the validation
strategy the team will follow (fit on train, tune on validation, evaluate the
champion once on the untouched test set).

PIPELINE POSITION
-----------------
    feature engineering -> [model proposal] -> approval -> training

Creating the proposal seeds the standing governance decisions and writes an
audit event, so the proposal is a first-class governed artifact.

OUTPUTS
-------
    governance/approvals/model_proposal.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import (
    ALPHA_GRID_SIZE,
    HIGH_PAIRWISE_CORRELATION,
    TARGET_COLUMN,
)
from config.paths import (
    ENGINEERED_TRAIN_PATH,
    MODEL_PROPOSAL_PATH,
    RESULTS_DIR,
    ensure_dir,
)
from src.governance import audit_logger, decision_tracker
from src.model_proposal.model_catalog import catalog_as_dicts, list_model_names
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PREDICTOR_CORRELATION_SCREEN: Path = (
    RESULTS_DIR / "eda_analysis_summary" / "predictor_correlation_screen.csv"
)


def _multicollinearity_flags() -> list[dict[str, Any]]:
    """
    Read the EDA predictor-correlation screen when present.

    Returns a list of high-pairwise-correlation flags (feature pair + value).
    Absence of the file is not an error; the proposal simply reports none.
    """

    if not PREDICTOR_CORRELATION_SCREEN.exists():
        return []

    screen = load_csv(PREDICTOR_CORRELATION_SCREEN)
    flags: list[dict[str, Any]] = []
    for _, row in screen.iterrows():
        absolute = float(row.get("absolute_correlation", 0.0) or 0.0)
        if absolute >= HIGH_PAIRWISE_CORRELATION:
            flags.append(
                {
                    "feature_1": row.get("feature_1"),
                    "feature_2": row.get("feature_2"),
                    "absolute_correlation": round(absolute, 4),
                }
            )
    return flags


def _inspect_training_data(train_df: pd.DataFrame) -> dict[str, Any]:
    """Summarise the engineered training frame for the proposal header."""

    predictors = [c for c in train_df.columns if c != TARGET_COLUMN]
    numeric_predictors = [
        c for c in predictors if pd.api.types.is_numeric_dtype(train_df[c].dtype)
    ]
    categorical_predictors = [c for c in predictors if c not in numeric_predictors]
    missing = {
        c: int(train_df[c].isna().sum())
        for c in train_df.columns
        if int(train_df[c].isna().sum()) > 0
    }
    return {
        "n_rows": int(len(train_df)),
        "n_predictors": len(predictors),
        "predictors": predictors,
        "numeric_predictors": numeric_predictors,
        "categorical_predictors": categorical_predictors,
        "columns_with_missing": missing,
        "target_column": TARGET_COLUMN,
    }


def build_proposal(train_df: pd.DataFrame) -> dict[str, Any]:
    """Assemble the full proposal record from the training data + catalogue."""

    data_summary = _inspect_training_data(train_df)
    flags = _multicollinearity_flags()
    recommended = list_model_names()

    collinear_note = (
        f"{len(flags)} predictor pair(s) exceed |r| >= {HIGH_PAIRWISE_CORRELATION}; "
        "this motivates the regularised Ridge and Lasso candidates alongside the "
        "plain LinearRegression baseline."
        if flags
        else "No high-correlation predictor pairs were flagged by the EDA screen."
    )

    validation_strategy = {
        "doctrine": "train / validation / test are kept strictly separated.",
        "fit": "All estimators and all fitted transforms (impute, encode, scale) "
        "are fit on the training split only.",
        "tune": "Ridge and Lasso alpha are tuned by scoring each candidate on the "
        f"dedicated validation split ({ALPHA_GRID_SIZE}-point log-spaced grid); "
        "k-fold over train+validation is intentionally NOT used.",
        "select": "The champion is the model with the best validation RMSE.",
        "final_test": "Only the champion is evaluated once on the untouched test "
        "split; the test split is never used for tuning or selection.",
        "metrics": ["rmse", "mae", "r2"],
    }

    proposal: dict[str, Any] = {
        "artifact": "model_proposal",
        "generated_at": utc_timestamp(),
        "modelling_scope": "linear family only (team decision; catalogue extensible)",
        "data_summary": data_summary,
        "multicollinearity_flags": flags,
        "multicollinearity_note": collinear_note,
        "recommended_models": recommended,
        "rationale": (
            "The problem is a continuous house-value regression assessed on "
            "interpretability and managerial defensibility. The linear family "
            "gives directly readable coefficients. LinearRegression is the "
            "transparent baseline; Ridge adds L2 shrinkage to stabilise the "
            "collinear count predictors; Lasso adds L1 selection to surface a "
            "parsimonious driver list for the executive briefing."
        ),
        "validation_strategy": validation_strategy,
        "candidate_catalog": catalog_as_dicts(),
    }
    return proposal


def write_proposal(
    train_df: pd.DataFrame | None = None,
    *,
    path: Path = MODEL_PROPOSAL_PATH,
) -> dict[str, Any]:
    """Build and persist the model proposal; seed decisions and audit-log it."""

    if train_df is None:
        train_df = load_csv(ENGINEERED_TRAIN_PATH)

    ensure_dir(path.parent)
    proposal = build_proposal(train_df)
    save_json(proposal, path)

    decision_tracker.seed_default_decisions()
    audit_logger.log_event(
        "model_proposal_created",
        {
            "recommended_models": proposal["recommended_models"],
            "n_rows": proposal["data_summary"]["n_rows"],
            "n_predictors": proposal["data_summary"]["n_predictors"],
            "multicollinearity_flags": len(proposal["multicollinearity_flags"]),
            "proposal_path": str(path),
        },
    )

    logger.info(
        "Model proposal written: %d recommended model(s), %d predictor(s) -> %s",
        len(proposal["recommended_models"]),
        proposal["data_summary"]["n_predictors"],
        path,
    )
    return proposal
