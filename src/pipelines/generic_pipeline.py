"""
src.pipelines.generic_pipeline
==============================

PURPOSE
-------
Prepare *any* registered dataset (built-in or uploaded via the GUI) for the
task-aware modelling layer, without a hand-written semantic schema. This is the
generalisation of the churn prep: it reads the active project's descriptor
(task / target / categoricals / drop / positive_class), cleans, splits
appropriately for the task, and writes the engineered splits + governance
approvals the shared ``train`` / ``evaluate`` stages consume.

    classification  -> stratified split on the target (preserves class balance)
    regression      -> plain random split

Run with the project active::

    WORKBENCH_PROJECT=<slug> python main.py autorun
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
from src.automl.detect import _coerce_numeric_like
from src.connectors.file_connector import load_local_csv
from src.governance import audit_logger, lineage_tracker
from src.model_proposal.model_catalog import MODEL_CATALOG, models_for_task
from src.preprocessing.splitter import split_by_target, split_random
from src.utils.common import utc_timestamp
from src.utils.file_utils import save_csv, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Coerce numeric-looking text columns and normalise the target's whitespace."""

    df = _coerce_numeric_like(df)
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = df[target].astype("string").str.strip()
    # Drop rows with a missing target (can't learn or score them).
    return df.dropna(subset=[target]).reset_index(drop=True)


def _write_governance(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Write feature + model approvals and the model proposal; return (predictors, models)."""

    d = active.active_dataset()
    target = d.target
    drop = set(d.drop_columns)
    predictor_columns = [c for c in df.columns if c != target and c not in drop]

    ensure_dir(FEATURE_APPROVAL_PATH.parent)
    save_json(
        {
            "artifact": "feature_approval",
            "generated_at": utc_timestamp(),
            "target_column": target,
            "dropped_columns": sorted(drop),
            "approved_predictor_columns": predictor_columns,
            "note": "Auto-approved: all non-identifier columns are predictors.",
        },
        FEATURE_APPROVAL_PATH,
    )

    models = models_for_task(d.task)
    save_json(
        {
            "artifact": "model_proposal",
            "generated_at": utc_timestamp(),
            "task": d.task,
            "selection_metric": d.selection_metric,
            "recommended_models": models,
            "candidate_catalog": [MODEL_CATALOG[m].as_dict() for m in models],
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
    return predictor_columns, models


def run_prep(prefer_cache: bool = True) -> dict:
    """Ingest (from workspace raw) -> clean -> split -> govern for the active project."""

    ensure_dirs()
    d = active.active_dataset()
    target = d.target
    task = d.task
    logger.info("=== Auto prep [%s | %s]: clean -> split -> govern ===", d.name, task)

    raw_path = DATA_RAW / d.file
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Upload the CSV for project "
            f"'{d.name}' before running autorun."
        )
    clean = _clean(load_local_csv(raw_path), target)

    if task == "classification":
        train_df, validation_df, test_df = split_by_target(clean, target)
    else:
        train_df, validation_df, test_df = split_random(clean)

    for frame, path in (
        (train_df, TRAIN_PATH), (validation_df, VALIDATION_PATH), (test_df, TEST_PATH),
        (train_df, ENGINEERED_TRAIN_PATH), (validation_df, ENGINEERED_VALIDATION_PATH),
        (test_df, ENGINEERED_TEST_PATH),
    ):
        ensure_dir(path.parent)
        save_csv(frame, path)

    predictors, models = _write_governance(clean)

    audit_logger.log_event(
        "auto_prep_run",
        {"project": d.name, "task": task, "rows": int(len(clean)), "n_predictors": len(predictors)},
    )
    lineage_tracker.record_node(
        "auto_prep",
        input_paths=[raw_path],
        output_paths=[ENGINEERED_TRAIN_PATH, ENGINEERED_VALIDATION_PATH, ENGINEERED_TEST_PATH],
        script="src/pipelines/generic_pipeline.py",
        params={"target": target, "task": task, "n_predictors": len(predictors)},
    )
    logger.info(
        "Auto prep complete: train=%d val=%d test=%d | %d predictors | models=%s",
        len(train_df), len(validation_df), len(test_df), len(predictors), models,
    )
    return {
        "rows": len(clean),
        "predictors": predictors,
        "models": models,
        "splits": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
    }


# ---------------------------------------------------------------------------
# The generic (uploaded-dataset) chain, defined ONCE.
#
# `main.py autorun` and `webapi/bridge.py analyze` both run this pipeline. They
# used to each keep their own hand-written list of calls, and those lists drifted:
# the web path generated the driver table and the HTML report, the CLI command
# did not, so the same dataset produced different deliverables depending on how
# it was launched.
#
# The bridge needs the phases separately, to report progress per stage; the CLI
# wants the whole thing. So the phases are named here and composed, rather than
# copied. Anything added to a phase is picked up by both callers.
# ---------------------------------------------------------------------------


def run_modelling() -> None:
    """Train every approved model, then compare and evaluate the champion."""

    from src.pipelines import evaluation_pipeline, training_pipeline

    training_pipeline.run()
    evaluation_pipeline.run()


def run_insights() -> None:
    """Every deliverable that follows evaluation.

    The trust audit and the HTML report are best-effort: both are deliverables,
    not preconditions, and a failure while producing either must not discard a
    completed analysis.
    """

    from src.automl import report as auto_report
    from src.explainability import drivers

    auto_report.run()
    # Returns None (never raises) when the champion exposes no per-feature
    # signal, which is a normal outcome for some model types.
    drivers.write_driver_artifacts()

    # After `drivers`, because the model card quotes the driver table when it
    # exists. Belongs in this phase rather than in a caller: the trust stage was
    # originally registered only with the workflow orchestrator, so `main.py all`
    # produced a model card and every uploaded dataset silently did not - exactly
    # the drift this module's header warns about.
    try:
        from src.pipelines import trust_pipeline

        trust_pipeline.run()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal.
        logger.warning("Trust audit skipped: %s", exc)

    try:
        from src.reporting import html_reporter

        html_reporter.write_report()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal.
        logger.warning("HTML project report skipped: %s", exc)


def run_full(prefer_cache: bool = True) -> dict:
    """Prep -> train -> evaluate -> every insight artifact. The whole chain."""

    summary = run_prep(prefer_cache=prefer_cache)
    run_modelling()
    run_insights()
    return summary
