"""
src.feature_engineering.feature_proposal
========================================

PURPOSE
-------
Create a formal Feature Engineering Plan from the training data and the schema.
This stage creates NO features; it only proposes them. Each proposal records:
which feature, why, its source variables, whether it is automatic or optional,
what validation is required before adoption, and the script that implements it.

PIPELINE POSITION
-----------------
    analytical summary -> [feature proposal] -> feature execution (automatic only)

Adapted from the professor's ``feature_engineering_plan.py``. The five feature
groups are preserved (Deterministic, Domain, Statistical, Machine Learning,
Inferential) along with the deterministic ratios and the leakage-risk /
training-fit metadata. Per the Milestone-1 brief, the Statistical group adds
log transforms of skewed count variables, and ONLY deterministic ratios and
those log transforms are flagged ``automatic=True``.

OUTPUTS
-------
    governance/approvals/feature_proposal.json   (governed approval artifact)
    results/feature_engineering/feature_plan.csv (tabular plan)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import APPROVALS_DIR, RESULTS_DIR, ensure_dir
from src.domain.schema import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN
from src.utils.common import utc_timestamp
from src.utils.file_utils import save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_RESULTS_DIR: Path = RESULTS_DIR / "feature_engineering"
FEATURE_PLAN_CSV: Path = FEATURE_RESULTS_DIR / "feature_plan.csv"
FEATURE_PROPOSAL_JSON: Path = APPROVALS_DIR / "feature_proposal.json"

# Skewed count variables that benefit from a log transform (log1p).
SKEWED_COUNT_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]

# Deterministic ratio definitions (numerator, denominator) from the professor's
# feature_engineering_plan.py.
RATIO_DEFINITIONS = [
    ("rooms_per_household", "total_rooms", "households", "Average housing-space availability per household."),
    ("bedrooms_per_room", "total_bedrooms", "total_rooms", "Bedroom share within the housing stock."),
    ("population_per_household", "population", "households", "Average household occupancy."),
    ("bedrooms_per_household", "total_bedrooms", "households", "Average bedroom availability per household."),
    ("rooms_per_person", "total_rooms", "population", "Housing-space availability per person."),
    ("bedrooms_per_person", "total_bedrooms", "population", "Bedroom availability per person."),
]


def _row(
    *,
    feature_name: str,
    feature_group: str,
    feature_subtype: str,
    source_columns: list[str],
    formula: str,
    business_meaning: str,
    rationale: str,
    automatic: bool,
    requires_training_fit: bool,
    validation_required: str,
    leakage_risk: str,
    implementing_script: str,
    priority: str,
    analyst_decision: str,
) -> dict[str, Any]:
    """Build one standardized proposal record."""

    return {
        "feature_name": feature_name,
        "feature_group": feature_group,
        "feature_subtype": feature_subtype,
        "source_columns": source_columns,
        "formula": formula,
        "business_meaning": business_meaning,
        "rationale": rationale,
        "automatic": automatic,
        "optional": not automatic,
        "requires_training_fit": requires_training_fit,
        "validation_required": validation_required,
        "leakage_risk": leakage_risk,
        "implementing_script": implementing_script,
        "priority": priority,
        "analyst_decision": analyst_decision,
    }


def _deterministic_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, numerator, denominator, meaning in RATIO_DEFINITIONS:
        if not {numerator, denominator}.issubset(data.columns):
            continue
        zero_count = int(
            (pd.to_numeric(data[denominator], errors="coerce") == 0).sum()
        )
        rows.append(
            _row(
                feature_name=name,
                feature_group="Deterministic",
                feature_subtype="Ratio",
                source_columns=[numerator, denominator],
                formula=f"{numerator} / {denominator}",
                business_meaning=meaning,
                rationale=(
                    "Separates observational-unit scale from density, "
                    "composition, or resource availability."
                ),
                automatic=True,
                requires_training_fit=False,
                validation_required=(
                    "Denominator must be non-zero (NaN on zero); preserve missing values."
                ),
                leakage_risk="LOW",
                implementing_script="src/feature_engineering/transformers.py",
                priority="HIGH",
                analyst_decision=(
                    "APPROVE FOR IMPLEMENTATION"
                    if zero_count == 0
                    else "APPROVE (safe-division handles zero denominators)"
                ),
            )
        )
    return rows


def _statistical_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Log transforms of skewed counts are deterministic and automatic.
    for column in SKEWED_COUNT_COLUMNS:
        if column not in data.columns:
            continue
        rows.append(
            _row(
                feature_name=f"log_{column}",
                feature_group="Statistical",
                feature_subtype="Log transform",
                source_columns=[column],
                formula=f"log1p({column})",
                business_meaning=f"Compresses the right-skewed scale of {column}.",
                rationale="EDA shows right-skewed count distributions; log1p stabilises scale.",
                automatic=True,
                requires_training_fit=False,
                validation_required="Input must be non-negative (log1p defined for >= 0).",
                leakage_risk="LOW",
                implementing_script="src/feature_engineering/transformers.py",
                priority="MEDIUM",
                analyst_decision="APPROVE FOR IMPLEMENTATION",
            )
        )
    # Interaction hypotheses remain optional (evaluate, do not impose).
    for name, sources, formula, meaning in [
        ("longitude_latitude_interaction", ["longitude", "latitude"], "longitude * latitude", "Joint geographic effect."),
        ("income_age_interaction", ["median_income", "housing_median_age"], "median_income * housing_median_age", "Income effect may vary with housing age."),
    ]:
        if not set(sources).issubset(data.columns):
            continue
        rows.append(
            _row(
                feature_name=name,
                feature_group="Statistical",
                feature_subtype="Interaction",
                source_columns=sources,
                formula=formula,
                business_meaning=meaning,
                rationale="Interaction hypothesis; not justified by marginal effects alone.",
                automatic=False,
                requires_training_fit=False,
                validation_required="Retain only with conceptual justification and validated value.",
                leakage_risk="LOW",
                implementing_script="src/feature_engineering/transformers.py",
                priority="MEDIUM",
                analyst_decision="EVALUATE, DO NOT IMPOSE",
            )
        )
    return rows


def _domain_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "ocean_proximity" in data.columns:
        rows.append(
            _row(
                feature_name="coastal_access_group",
                feature_group="Domain",
                feature_subtype="Category consolidation",
                source_columns=["ocean_proximity"],
                formula="Business-approved grouping of proximity categories",
                business_meaning="Simplified market-access / coastal-amenity segment.",
                rationale="May improve interpretability when individual categories are sparse.",
                automatic=False,
                requires_training_fit=False,
                validation_required="Grouping must be approved and applied identically to every split.",
                leakage_risk="LOW",
                implementing_script="src/feature_engineering/transformers.py",
                priority="LOW",
                analyst_decision="REQUIRES DOMAIN APPROVAL",
            )
        )
    return rows


def _ml_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            feature_name="standardized_numeric_predictors",
            feature_group="Machine Learning",
            feature_subtype="Scaling",
            source_columns=[c for c in NUMERICAL_COLUMNS if c in data.columns and c != TARGET_COLUMN],
            formula="StandardScaler fitted on training data",
            business_meaning="Places numerical predictors on comparable scales.",
            rationale="Required for stable gradient-based optimisation.",
            automatic=False,  # handled by the preprocessing scaler, not this stage.
            requires_training_fit=True,
            validation_required="Fit on training data only; reuse for validation/test.",
            leakage_risk="HIGH",
            implementing_script="src/preprocessing/scaler.py",
            priority="HIGH",
            analyst_decision="APPROVED (executed in preprocessing scaler)",
        )
    )
    if CATEGORICAL_COLUMNS:
        rows.append(
            _row(
                feature_name="one_hot_categorical_predictors",
                feature_group="Machine Learning",
                feature_subtype="Categorical encoding",
                source_columns=[c for c in CATEGORICAL_COLUMNS if c in data.columns],
                formula="OneHotEncoder(handle_unknown='ignore') fitted on training data",
                business_meaning="Represents categorical differences without imposing order.",
                rationale="Required by linear and gradient-based estimators.",
                automatic=False,  # handled by the preprocessing encoder.
                requires_training_fit=True,
                validation_required="Fit category levels on training data only.",
                leakage_risk="HIGH",
                implementing_script="src/preprocessing/encoder.py",
                priority="HIGH",
                analyst_decision="APPROVED (executed in preprocessing encoder)",
            )
        )
    return rows


def _inferential_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        _row(
            feature_name="centered_numeric_predictors",
            feature_group="Inferential",
            feature_subtype="Centering",
            source_columns=[c for c in NUMERICAL_COLUMNS if c in data.columns and c != TARGET_COLUMN],
            formula="x - training mean",
            business_meaning="Makes the intercept and interaction terms interpretable.",
            rationale="Reduces nonessential collinearity in polynomial/interaction specs.",
            automatic=False,
            requires_training_fit=True,
            validation_required="Estimate centering constants from training data only.",
            leakage_risk="HIGH",
            implementing_script="src/feature_engineering/transformers.py",
            priority="MEDIUM",
            analyst_decision="USE WHEN POLYNOMIAL OR INTERACTION TERMS ENTER",
        )
    ]


def build_proposal(train_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Assemble the full ordered proposal list across all five groups."""

    rows: list[dict[str, Any]] = []
    rows.extend(_deterministic_rows(train_df))
    rows.extend(_domain_rows(train_df))
    rows.extend(_statistical_rows(train_df))
    rows.extend(_ml_rows(train_df))
    rows.extend(_inferential_rows(train_df))
    for index, row in enumerate(rows, start=1):
        row["feature_id"] = f"FEAT-{index:03d}"
    return rows


def write_proposal(
    train_df: pd.DataFrame,
    csv_path: Path = FEATURE_PLAN_CSV,
    json_path: Path = FEATURE_PROPOSAL_JSON,
) -> list[dict[str, Any]]:
    """Build and persist the proposal (governance JSON + results CSV)."""

    ensure_dir(csv_path.parent)
    ensure_dir(json_path.parent)

    proposal = build_proposal(train_df)

    # CSV view flattens the source-column list for readability.
    plan_df = pd.DataFrame(proposal)
    plan_df["source_columns"] = plan_df["source_columns"].apply(lambda cols: " | ".join(cols))
    ordered_columns = [
        "feature_id", "feature_name", "feature_group", "feature_subtype",
        "source_columns", "formula", "business_meaning", "rationale",
        "automatic", "optional", "requires_training_fit", "validation_required",
        "leakage_risk", "implementing_script", "priority", "analyst_decision",
    ]
    save_csv(plan_df[ordered_columns], csv_path)

    automatic_features = [row["feature_name"] for row in proposal if row["automatic"]]
    governance_record = {
        "artifact": "feature_proposal",
        "generated_at": utc_timestamp(),
        "target_column": TARGET_COLUMN,
        "training_observations": int(len(train_df)),
        "total_proposed": len(proposal),
        "automatic_features": automatic_features,
        "automatic_count": len(automatic_features),
        "note": (
            "Only features with automatic=True are executed by "
            "src/feature_engineering/pipeline.py. Optional proposals require "
            "analyst/domain approval before execution."
        ),
        "features": proposal,
    }
    save_json(governance_record, json_path)

    logger.info(
        "Feature proposal written: %d proposed, %d automatic -> %s / %s",
        len(proposal), len(automatic_features), csv_path, json_path,
    )
    return proposal
