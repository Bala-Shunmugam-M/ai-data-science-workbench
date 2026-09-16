"""
src.pipelines.eda_pipeline
==========================

PURPOSE
-------
Data understanding + EDA + analytical summary on the TRAINING split only.

PIPELINE POSITION
-----------------
Stage 3 (deep) Data Understanding + Stage on EDA. The test set is never read.

WHY THIS STAGE REFUSES SOME PROJECTS
------------------------------------
Every table here is driven by ``src.domain.schema.SEMANTIC_SCHEMA``, which is the
human-approved contract for the housing dataset — the stage reports on the columns
the schema names, not on whatever the frame happens to contain. Point it at a
different project (churn, an uploaded CSV) and the schema names none of its
columns.

Unguarded, that surfaced as ``KeyError: 'ocean_proximity'`` or pandas'
``Cannot describe a DataFrame without columns`` from three frames deep, which reads
as a broken pipeline. It is not: it is a dataset the governed schema does not
describe. :func:`_require_schema_coverage` says exactly that instead, and a
partially-covered frame still gets tables for the columns the schema does name.
"""

from __future__ import annotations

import pandas as pd

from config.paths import TRAIN_PATH, ensure_dirs
from src.domain.schema import schema_columns
from src.eda.analysis_summary import run_analysis_summary
from src.eda.explorer import run_eda
from src.eda.understanding import run_understanding
from src.utils.file_utils import load_csv
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _require_schema_coverage(train_df: pd.DataFrame) -> None:
    """Fail with the real reason when the governed schema describes none of the data."""

    covered = [column for column in schema_columns() if column in train_df.columns]
    if covered:
        logger.info(
            "Semantic schema covers %d of %d training columns.",
            len(covered),
            len(train_df.columns),
        )
        return

    raise ValueError(
        "The semantic schema (src/domain/schema.py) describes none of this dataset's "
        f"{len(train_df.columns)} columns, so schema-driven EDA has nothing to report. "
        "This stage is governed by that schema by design - it is not a dataset-agnostic "
        "profiler. Either add this dataset to the schema, or use the shorter pipeline "
        "(churn-prep / autorun), which skips this stage for exactly this reason."
    )


def run_understand() -> None:
    """Data-understanding sub-stage."""

    # Coverage is checked BEFORE ensure_dirs: a refused stage must leave nothing
    # behind. An empty results/eda/ is indistinguishable from a finished one to
    # anything that probes the filesystem, the web app included.
    train_df = load_csv(TRAIN_PATH)
    _require_schema_coverage(train_df)
    ensure_dirs()
    logger.info("=== Data understanding ===")
    run_understanding(train_df)


def run_eda_stage() -> None:
    """EDA + analytical-summary sub-stage."""

    train_df = load_csv(TRAIN_PATH)
    _require_schema_coverage(train_df)
    ensure_dirs()
    logger.info("=== EDA + analytical summary ===")
    eda_outputs = run_eda(train_df)
    run_analysis_summary(train_df, eda_outputs)


def run() -> None:
    """Full understanding + EDA + summary stage."""

    run_understand()
    run_eda_stage()
    logger.info("EDA stage complete.")
