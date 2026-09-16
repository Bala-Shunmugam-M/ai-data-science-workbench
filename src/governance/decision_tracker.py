"""
src.governance.decision_tracker
===============================

PURPOSE
-------
A small registry of human-readable governance decisions (decision, rationale,
date). It records the standing team choices that shape the whole pipeline -
e.g. "linear family only" and "median imputation after split" - so a reviewer
can see *why* the platform behaves as it does without reading the code.

PIPELINE POSITION
-----------------
Seeded during model proposal and readable by the reporting stage. Decisions are
keyed by a stable slug so re-running the pipeline updates in place rather than
duplicating entries.

OUTPUTS
-------
    governance/decisions.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.paths import DECISIONS_PATH, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Standing decisions carried over from the plan and the professor's doctrine.
DEFAULT_DECISIONS: list[dict[str, str]] = [
    {
        "key": "linear_family_only",
        "decision": "Restrict the model catalogue to the linear family "
        "(LinearRegression, Ridge, Lasso).",
        "rationale": "Team decision favouring transparent, defensible, "
        "coefficient-interpretable models for a managerial audience; the "
        "catalogue stays extensible so ensembles can be added later.",
    },
    {
        "key": "split_early_test_untouched",
        "decision": "Split into train/validation/test early; the test split is "
        "untouched until a single final evaluation of the champion.",
        "rationale": "Professor's explicit doctrine; prevents optimistic bias "
        "and information leakage into model selection.",
    },
    {
        "key": "median_imputation_after_split",
        "decision": "Impute missing total_bedrooms with the median fitted on "
        "the training split only, after the split.",
        "rationale": "Prevents leakage of validation/test information into the "
        "imputation statistic.",
    },
    {
        "key": "tune_on_validation_not_kfold",
        "decision": "Tune Ridge/Lasso alpha on the dedicated validation split "
        "rather than k-fold cross-validation over train+validation.",
        "rationale": "Respects the professor's train/validation/test doctrine; "
        "keeps selection and final testing strictly separated.",
    },
    {
        "key": "governed_approvals_gate_training",
        "decision": "Training refuses any model not present in "
        "model_approval.json; features not approved are excluded from the "
        "modelling matrix.",
        "rationale": "Enforces the approval workflow so only governed, "
        "human-approved artifacts reach production.",
    },
]


def record_decision(
    key: str,
    decision: str,
    rationale: str,
    *,
    path: Path = DECISIONS_PATH,
) -> dict[str, Any]:
    """Insert or update one decision (keyed by ``key``) and persist the registry."""

    registry = _load(path)
    entry = {
        "key": key,
        "decision": decision,
        "rationale": rationale,
        "date": utc_timestamp(),
    }
    registry[key] = entry
    _save(registry, path)
    logger.info("Recorded governance decision '%s'.", key)
    return entry


def seed_default_decisions(path: Path = DECISIONS_PATH) -> list[dict[str, Any]]:
    """Ensure the standing team decisions are present; return all decisions."""

    registry = _load(path)
    for item in DEFAULT_DECISIONS:
        if item["key"] not in registry:
            registry[item["key"]] = {
                **item,
                "date": utc_timestamp(),
            }
    _save(registry, path)
    return list(registry.values())


def list_decisions(path: Path = DECISIONS_PATH) -> list[dict[str, Any]]:
    """Return all recorded decisions (empty list when none exist yet)."""

    return list(_load(path).values())


def _load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    # Stored as {"decisions": {key: entry}} for a stable top-level shape.
    return dict(data.get("decisions", {})) if isinstance(data, dict) else {}


def _save(registry: dict[str, dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    save_json({"decisions": registry}, path)
