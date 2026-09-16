"""
src.governance.approvals
========================

PURPOSE
-------
The approval workflow that gates modelling. Two functions read their upstream
proposal JSON, mark entries approved (all recommended/automatic ones by default,
or an explicit subset), persist an approval artifact, and audit-log the
decision:

    approve_features()  reads feature_proposal.json  -> feature_approval.json
    approve_models()    reads model_proposal.json    -> model_approval.json

Training then consults ``model_approval.json`` via :func:`require_model_approved`
and refuses (``GovernanceError``) to train any model that is not approved. It
also exposes the approved predictor columns so the trainer can exclude the
target and any non-approved column from the modelling matrix.

PIPELINE POSITION
-----------------
    proposal -> [approval] -> training

OUTPUTS
-------
    governance/approvals/feature_approval.json
    governance/approvals/model_approval.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from config.constants import TARGET_COLUMN
from config.paths import (
    ENGINEERED_TRAIN_PATH,
    FEATURE_APPROVAL_PATH,
    FEATURE_PROPOSAL_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_PROPOSAL_PATH,
    ensure_dir,
)
from src.domain.schema import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from src.governance import GovernanceError, audit_logger
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_csv, load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Base schema predictors are always eligible for modelling once approved; the
# governed choice is which *engineered* features to add on top.
_BASE_NUMERIC_PREDICTORS = [c for c in NUMERICAL_COLUMNS if c != TARGET_COLUMN]
_BASE_CATEGORICAL_PREDICTORS = list(CATEGORICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Feature approval.
# ---------------------------------------------------------------------------
def approve_features(
    subset: Iterable[str] | None = None,
    *,
    proposal_path: Path = FEATURE_PROPOSAL_PATH,
    approval_path: Path = FEATURE_APPROVAL_PATH,
    engineered_path: Path = ENGINEERED_TRAIN_PATH,
    actor: str | None = None,
) -> dict[str, Any]:
    """
    Approve engineered features and write ``feature_approval.json``.

    By default every ``automatic=True`` proposal is approved. Passing ``subset``
    restricts approval to those feature names. The approved predictor columns
    (base schema predictors + approved engineered features that actually exist
    in the engineered training data) are recorded for the trainer's guard.
    """

    proposal = load_json(proposal_path)
    features: list[dict[str, Any]] = proposal.get("features", [])

    automatic = [f["feature_name"] for f in features if f.get("automatic")]
    if subset is None:
        chosen = automatic
    else:
        requested = list(subset)
        known = {f["feature_name"] for f in features}
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError(f"Unknown feature(s) in subset: {unknown}")
        chosen = requested

    # Restrict approved engineered features to those present in engineered data.
    present_columns: set[str] = set()
    if engineered_path.exists():
        present_columns = set(load_csv(engineered_path, nrows=1).columns)

    approved_engineered = [f for f in chosen if not present_columns or f in present_columns]
    approved_predictor_columns = (
        [c for c in _BASE_NUMERIC_PREDICTORS if not present_columns or c in present_columns]
        + [c for c in _BASE_CATEGORICAL_PREDICTORS if not present_columns or c in present_columns]
        + approved_engineered
    )

    marked = [
        {**f, "approved": f["feature_name"] in chosen}
        for f in features
    ]

    record: dict[str, Any] = {
        "artifact": "feature_approval",
        "generated_at": utc_timestamp(),
        "target_column": TARGET_COLUMN,
        "approved_features": [f["feature_name"] for f in marked if f["approved"]],
        "approved_count": sum(1 for f in marked if f["approved"]),
        "base_predictor_columns": _BASE_NUMERIC_PREDICTORS + _BASE_CATEGORICAL_PREDICTORS,
        "approved_engineered_features": approved_engineered,
        "approved_predictor_columns": approved_predictor_columns,
        "features": marked,
    }
    ensure_dir(approval_path.parent)
    save_json(record, approval_path)

    audit_logger.log_event(
        "feature_approval_granted",
        {
            "approved_count": record["approved_count"],
            "approved_predictor_columns": approved_predictor_columns,
            "approval_path": str(approval_path),
        },
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    logger.info(
        "Feature approval written: %d feature(s), %d predictor column(s) -> %s",
        record["approved_count"],
        len(approved_predictor_columns),
        approval_path,
    )
    return record


# ---------------------------------------------------------------------------
# Model approval.
# ---------------------------------------------------------------------------
def approve_models(
    subset: Iterable[str] | None = None,
    *,
    proposal_path: Path = MODEL_PROPOSAL_PATH,
    approval_path: Path = MODEL_APPROVAL_PATH,
    actor: str | None = None,
) -> dict[str, Any]:
    """
    Approve candidate models and write ``model_approval.json``.

    By default every recommended model in the proposal is approved. Passing
    ``subset`` restricts approval to those model names.
    """

    proposal = load_json(proposal_path)
    recommended: list[str] = proposal.get("recommended_models", [])

    if subset is None:
        chosen = recommended
    else:
        requested = list(subset)
        known = {m["name"] for m in proposal.get("candidate_catalog", [])} or set(recommended)
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise ValueError(f"Unknown model(s) in subset: {unknown}")
        chosen = requested

    record: dict[str, Any] = {
        "artifact": "model_approval",
        "generated_at": utc_timestamp(),
        "approved_models": chosen,
        "approved_count": len(chosen),
        "recommended_models": recommended,
        "note": "Training refuses any model not listed in approved_models.",
    }
    ensure_dir(approval_path.parent)
    save_json(record, approval_path)

    audit_logger.log_event(
        "model_approval_granted",
        {"approved_models": chosen, "approval_path": str(approval_path)},
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    logger.info("Model approval written: %s -> %s", chosen, approval_path)
    return record


# ---------------------------------------------------------------------------
# Read helpers + the training guard.
# ---------------------------------------------------------------------------
def load_model_approval(path: Path = MODEL_APPROVAL_PATH) -> dict[str, Any]:
    """Return the model-approval record, or raise ``GovernanceError`` if absent."""

    if not path.exists():
        raise GovernanceError(
            "No model_approval.json found. Run 'python main.py approve --models' "
            "before training."
        )
    return load_json(path)


def approved_models(path: Path = MODEL_APPROVAL_PATH) -> list[str]:
    """Return the list of approved model names (empty when none)."""

    if not path.exists():
        return []
    return list(load_json(path).get("approved_models", []))


def require_model_approved(name: str, *, path: Path = MODEL_APPROVAL_PATH) -> None:
    """
    Guard: raise ``GovernanceError`` unless ``name`` is an approved model.

    This is the enforcement point that makes the approval workflow binding:
    training MUST call it before fitting any estimator.
    """

    approved = approved_models(path)
    if name not in approved:
        raise GovernanceError(
            f"Model '{name}' is not approved for training. "
            f"Approved models: {approved or '[]'}. "
            "Approve it via 'python main.py approve --models' first."
        )


def load_feature_approval(path: Path = FEATURE_APPROVAL_PATH) -> dict[str, Any]:
    """Return the feature-approval record, or raise ``GovernanceError`` if absent."""

    if not path.exists():
        raise GovernanceError(
            "No feature_approval.json found. Run 'python main.py approve --features' "
            "before training."
        )
    return load_json(path)


def approved_predictor_columns(path: Path = FEATURE_APPROVAL_PATH) -> list[str]:
    """Return the governed list of predictor columns approved for modelling."""

    return list(load_feature_approval(path).get("approved_predictor_columns", []))
