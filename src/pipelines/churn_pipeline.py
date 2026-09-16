"""
src.pipelines.churn_pipeline
============================

PURPOSE
-------
Prepare the Customer Churn (classification) project end to end up to the point
the *generalised* modelling layer takes over. The housing ingestion ->
feature-engineering stages are bound to the housing semantic schema, so the
churn project uses this dedicated prep instead, then reuses the SAME task-aware
``train`` and ``evaluate`` stages.

Steps
-----
1. Ingest the Telco churn CSV into the churn workspace's ``data/raw``.
2. Clean the one known quirk: ``TotalCharges`` has blank strings for
   tenure-0 customers; coerce it to numeric (blanks -> NaN, imputed later by the
   modelling preprocessor, fit on train only).
3. Stratified 70/15/15 split on the target (preserves the ~26.5% churn rate).
4. Write the "engineered" splits (identical to the cleaned splits: churn needs
   no housing-style feature engineering) plus the governance approval artifacts
   the trainer's guard consumes.

Run with the churn workspace active::

    WORKBENCH_PROJECT=churn python main.py churn-prep

PIPELINE POSITION
-----------------
    [churn-prep] -> train -> evaluate -> (retention simulator)
"""

from __future__ import annotations

import pandas as pd

from config import active
from config.paths import (
    DATA_RAW,
    ENGINEERED_TEST_PATH,
    ENGINEERED_TRAIN_PATH,
    ENGINEERED_VALIDATION_PATH,
    FEATURE_APPROVAL_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_PROPOSAL_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    ensure_dir,
    ensure_dirs,
)
from src.connectors.file_connector import load_local_csv
from src.connectors.kaggle_connector import download_kaggle_dataset
from src.governance import audit_logger, lineage_tracker
from src.model_proposal.model_catalog import MODEL_CATALOG, models_for_task
from src.preprocessing.splitter import split_by_target
from src.utils.common import utc_timestamp
from src.utils.file_utils import save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _ingest(prefer_cache: bool = True) -> pd.DataFrame:
    """Download (or reuse the cached) churn CSV into the churn workspace."""

    descriptor = active.active_dataset()
    raw_path = DATA_RAW / descriptor.file
    if prefer_cache and raw_path.exists():
        logger.info("Loading cached churn raw data from %s", raw_path)
        return load_local_csv(raw_path)
    return download_kaggle_dataset(descriptor.kaggle_id, descriptor.file, raw_path)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges to numeric and normalise the target's whitespace."""

    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    target = active.target_column()
    df[target] = df[target].astype("string").str.strip()

    # Match `generic_pipeline._clean`: a row with no label cannot be trained on
    # or scored, and stratifying on NaN produces a phantom class. The Telco CSV
    # has no missing labels today, so this is a no-op there - but the two prep
    # paths had drifted, and the churn one would have silently carried nulls
    # through into the split.
    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < before:
        logger.warning(
            "Dropped %d row(s) with a missing '%s' label.", before - len(df), target
        )
    return df


def _write_governance(df: pd.DataFrame) -> list[str]:
    """Write feature + model approvals and the model proposal for churn."""

    descriptor = active.active_dataset()
    target = descriptor.target
    drop = set(descriptor.drop_columns)
    predictor_columns = [c for c in df.columns if c != target and c not in drop]

    ensure_dir(FEATURE_APPROVAL_PATH.parent)
    save_json(
        {
            "artifact": "feature_approval",
            "generated_at": utc_timestamp(),
            "target_column": target,
            "dropped_columns": sorted(drop),
            "approved_predictor_columns": predictor_columns,
            "note": "Churn uses all non-identifier columns as predictors.",
        },
        FEATURE_APPROVAL_PATH,
    )

    models = models_for_task("classification")
    save_json(
        {
            "artifact": "model_proposal",
            "generated_at": utc_timestamp(),
            "task": "classification",
            "selection_metric": descriptor.selection_metric,
            "recommended_models": models,
            "candidate_catalog": [
                MODEL_CATALOG[m].as_dict() for m in models
            ],
        },
        MODEL_PROPOSAL_PATH,
    )
    save_json(
        {
            "artifact": "model_approval",
            "generated_at": utc_timestamp(),
            "approved_models": models,
            "approved_count": len(models),
            "recommended_models": models,
            "note": "Training refuses any model not listed in approved_models.",
        },
        MODEL_APPROVAL_PATH,
    )
    return predictor_columns


def run_prep(prefer_cache: bool = True) -> dict:
    """Run the full churn prep and return a small summary dict."""

    if not active.is_classification():
        raise RuntimeError(
            "churn-prep is only valid for a classification project. "
            "Run it with WORKBENCH_PROJECT=churn."
        )

    ensure_dirs()
    target = active.target_column()
    logger.info("=== Churn prep: ingest -> clean -> split -> govern ===")

    raw = _ingest(prefer_cache=prefer_cache)
    clean = _clean(raw)
    logger.info("Churn data cleaned: shape=%s | churn rate=%.1f%%",
                clean.shape, 100 * (clean[target] == active.positive_class()).mean())

    train_df, validation_df, test_df = split_by_target(clean, target)
    for frame, path in (
        (train_df, TRAIN_PATH), (validation_df, VALIDATION_PATH), (test_df, TEST_PATH),
        (train_df, ENGINEERED_TRAIN_PATH), (validation_df, ENGINEERED_VALIDATION_PATH),
        (test_df, ENGINEERED_TEST_PATH),
    ):
        ensure_dir(path.parent)
        save_csv(frame, path)

    predictors = _write_governance(clean)

    audit_logger.log_event(
        "churn_prep_run",
        {
            "rows": int(len(clean)),
            "n_predictors": len(predictors),
            "splits": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        },
    )
    lineage_tracker.record_node(
        "churn_prep",
        input_paths=[DATA_RAW / active.active_dataset().file],
        output_paths=[ENGINEERED_TRAIN_PATH, ENGINEERED_VALIDATION_PATH, ENGINEERED_TEST_PATH],
        script="src/pipelines/churn_pipeline.py",
        params={"target": target, "n_predictors": len(predictors)},
    )

    logger.info(
        "Churn prep complete: train=%d val=%d test=%d | %d predictors",
        len(train_df), len(validation_df), len(test_df), len(predictors),
    )
    return {
        "rows": len(clean),
        "predictors": predictors,
        "splits": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
    }
