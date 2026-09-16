"""
config.paths
============

PURPOSE
-------
Single source of truth for every directory the workbench reads or writes.

PIPELINE POSITION
-----------------
Imported by every stage (ingestion -> validation -> preprocessing -> split ->
data understanding -> EDA -> feature engineering). Centralising the paths here
guarantees that all stages agree on where raw data, processed data, splits,
engineered features, models, artifacts, results, and governance approvals live.

All locations are derived from ``PROJECT_ROOT`` via :mod:`pathlib`, so the tree
is portable across machines and operating systems.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Root anchor: config/paths.py -> config/ -> project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "config"

# ---------------------------------------------------------------------------
# Active project + its workspace root.
#
# The active project is chosen by the ``WORKBENCH_PROJECT`` environment
# variable (default: the graded California Housing showcase). The showcase keeps
# the legacy flat layout at PROJECT_ROOT because its model registry stores
# absolute model paths and its graded report already lives there. EVERY OTHER
# project is isolated under ``workspaces/<slug>/`` so projects never overwrite
# one another's data, models, or artifacts.
# ---------------------------------------------------------------------------
_HOUSING_ALIASES = {"", "california-housing", "california_housing"}
ACTIVE_PROJECT: str = (
    os.environ.get("WORKBENCH_PROJECT", "california-housing").strip()
    or "california-housing"
)
WORKSPACE_ROOT: Path = (
    PROJECT_ROOT
    if ACTIVE_PROJECT in _HOUSING_ALIASES
    else PROJECT_ROOT / "workspaces" / ACTIVE_PROJECT
)

# ---------------------------------------------------------------------------
# Data lake (stage-by-stage datasets).
# ---------------------------------------------------------------------------
DATA_DIR: Path = WORKSPACE_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_SPLITS: Path = DATA_DIR / "splits"
DATA_ENGINEERED: Path = DATA_DIR / "engineered"

# ---------------------------------------------------------------------------
# Model store, generated artifacts, analytical results, governance approvals.
# ---------------------------------------------------------------------------
MODELS_DIR: Path = WORKSPACE_ROOT / "models"
ARTIFACTS_DIR: Path = WORKSPACE_ROOT / "artifacts"
RESULTS_DIR: Path = WORKSPACE_ROOT / "results"
GOVERNANCE_DIR: Path = WORKSPACE_ROOT / "governance"

# ---------------------------------------------------------------------------
# Common sub-locations used across stages.
# ---------------------------------------------------------------------------
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
APPROVALS_DIR: Path = GOVERNANCE_DIR / "approvals"

# ---------------------------------------------------------------------------
# Milestone 2: governance sub-trees (audit trail, lineage, decisions).
# ---------------------------------------------------------------------------
AUDIT_DIR: Path = GOVERNANCE_DIR / "audit"
LINEAGE_DIR: Path = GOVERNANCE_DIR / "lineage"
AUDIT_LOG_PATH: Path = AUDIT_DIR / "audit_log.jsonl"
LINEAGE_PATH: Path = LINEAGE_DIR / "lineage.json"
DECISIONS_PATH: Path = GOVERNANCE_DIR / "decisions.json"

# Governed proposal / approval artifacts (Milestone 2).
MODEL_PROPOSAL_PATH: Path = APPROVALS_DIR / "model_proposal.json"
FEATURE_APPROVAL_PATH: Path = APPROVALS_DIR / "feature_approval.json"
MODEL_APPROVAL_PATH: Path = APPROVALS_DIR / "model_approval.json"
FEATURE_PROPOSAL_PATH: Path = APPROVALS_DIR / "feature_proposal.json"

# ---------------------------------------------------------------------------
# Milestone 2: model store, registry, evaluation, reports, experiments, store.
# ---------------------------------------------------------------------------
MODEL_REGISTRY_PATH: Path = MODELS_DIR / "model_registry.json"

EVALUATION_DIR: Path = ARTIFACTS_DIR / "evaluation"
EVALUATION_FIGURES_DIR: Path = EVALUATION_DIR / "figures"
MODEL_COMPARISON_CSV: Path = EVALUATION_DIR / "model_comparison.csv"
MODEL_EVALUATION_REPORT_PATH: Path = EVALUATION_DIR / "model_evaluation_report.json"
FINAL_MODEL_SELECTION_PATH: Path = ARTIFACTS_DIR / "final_model_selection.json"

REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
EXECUTIVE_BRIEFING_PATH: Path = REPORTS_DIR / "executive_briefing.md"
PROJECT_REPORT_PATH: Path = REPORTS_DIR / "project_report.html"

EXPERIMENTS_DIR: Path = ARTIFACTS_DIR / "experiments"
EXPERIMENTS_LOG_PATH: Path = EXPERIMENTS_DIR / "experiments.jsonl"

ARTIFACT_STORE_DIR: Path = ARTIFACTS_DIR / "store"

EXPLAINABILITY_DIR: Path = ARTIFACTS_DIR / "explainability"

# ---------------------------------------------------------------------------
# Trust layer: the audit-grade evidence a governed model is supposed to ship
# with - subgroup fairness, model-agnostic importance, probability calibration,
# and the model card that assembles them. Written by the ``trust`` stage, which
# runs after ``evaluate`` has frozen the champion.
# ---------------------------------------------------------------------------
TRUST_DIR: Path = ARTIFACTS_DIR / "trust"
TRUST_FIGURES_DIR: Path = TRUST_DIR / "figures"

MODEL_CARD_MD_PATH: Path = TRUST_DIR / "model_card.md"
MODEL_CARD_JSON_PATH: Path = TRUST_DIR / "model_card.json"
FAIRNESS_JSON_PATH: Path = TRUST_DIR / "subgroup_fairness.json"
FAIRNESS_CSV_PATH: Path = TRUST_DIR / "subgroup_fairness.csv"
ROBUST_IMPORTANCE_JSON_PATH: Path = TRUST_DIR / "permutation_importance.json"
ROBUST_IMPORTANCE_CSV_PATH: Path = TRUST_DIR / "permutation_importance.csv"
CALIBRATION_JSON_PATH: Path = TRUST_DIR / "calibration.json"

# ---------------------------------------------------------------------------
# Milestone 3: workflow orchestrator status.
# ---------------------------------------------------------------------------
WORKFLOW_STATUS_PATH: Path = ARTIFACTS_DIR / "workflow_status.json"

# Canonical dataset file names for the California Housing showcase project.
RAW_HOUSING_PATH: Path = DATA_RAW / "housing.csv"
PROCESSED_HOUSING_PATH: Path = DATA_PROCESSED / "housing_clean.csv"
TRAIN_PATH: Path = DATA_SPLITS / "train.csv"
VALIDATION_PATH: Path = DATA_SPLITS / "validation.csv"
TEST_PATH: Path = DATA_SPLITS / "test.csv"

ENGINEERED_TRAIN_PATH: Path = DATA_ENGINEERED / "train.csv"
ENGINEERED_VALIDATION_PATH: Path = DATA_ENGINEERED / "validation.csv"
ENGINEERED_TEST_PATH: Path = DATA_ENGINEERED / "test.csv"

# Every top-level directory that must exist for a clean run.
ALL_DIRECTORIES: tuple[Path, ...] = (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_SPLITS,
    DATA_ENGINEERED,
    MODELS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    GOVERNANCE_DIR,
    LOGS_DIR,
    APPROVALS_DIR,
    AUDIT_DIR,
    LINEAGE_DIR,
    EVALUATION_DIR,
    EVALUATION_FIGURES_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    ARTIFACT_STORE_DIR,
    EXPLAINABILITY_DIR,
    TRUST_DIR,
    TRUST_FIGURES_DIR,
)


def ensure_dir(directory: Path) -> Path:
    """Create ``directory`` (and parents) if missing and return it."""

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_dirs(directories: Iterable[Path] | None = None) -> None:
    """
    Create the given directories, or the full workbench tree when ``None``.

    Stages call this before writing so a fresh clone can run end-to-end.
    """

    for directory in (directories if directories is not None else ALL_DIRECTORIES):
        ensure_dir(directory)
