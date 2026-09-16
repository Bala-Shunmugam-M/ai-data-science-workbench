"""
app.pages.2_Data
=================

PURPOSE
-------
Raw/processed/split preview tables, the human-approved semantic schema, and
the semantic-validation report.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import project_state  # noqa: E402

_PROJECT = project_state.apply_project()

from app.common import PAGE_ICON, not_run_yet, page_header, safe_read_csv, safe_read_text  # noqa: E402
from config.paths import (  # noqa: E402
    PROCESSED_HOUSING_PATH,
    RAW_HOUSING_PATH,
    RESULTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from src.domain.schema import SEMANTIC_SCHEMA  # noqa: E402

st.set_page_config(page_title="Data - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Data", "Raw, processed, and split previews; schema; validation report.")

# ---------------------------------------------------------------------------
# Raw / processed / splits previews.
# ---------------------------------------------------------------------------
st.subheader("Dataset previews")
tabs = st.tabs(["Raw", "Processed", "Train", "Validation", "Test"])
for tab, path, label, cmd in [
    (tabs[0], RAW_HOUSING_PATH, "Raw dataset", "ingest"),
    (tabs[1], PROCESSED_HOUSING_PATH, "Processed dataset", "preprocess"),
    (tabs[2], TRAIN_PATH, "Train split", "preprocess"),
    (tabs[3], VALIDATION_PATH, "Validation split", "preprocess"),
    (tabs[4], TEST_PATH, "Test split", "preprocess"),
]:
    with tab:
        df = safe_read_csv(path)
        if df is None:
            not_run_yet(label, cmd)
        else:
            st.caption(f"{len(df):,} rows x {df.shape[1]} columns - {path.name}")
            st.dataframe(df.head(200), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Semantic schema table (human-approved contract).
# ---------------------------------------------------------------------------
st.subheader("Human-approved semantic schema")
schema_rows = []
for column, rules in SEMANTIC_SCHEMA.items():
    schema_rows.append(
        {
            "column": column,
            "role": rules.get("role"),
            "semantic_type": rules.get("semantic_type"),
            "expected_storage_type": rules.get("expected_storage_type"),
            "nullable": rules.get("nullable"),
            "minimum": rules.get("minimum"),
            "maximum": rules.get("maximum"),
            "whole_number": rules.get("whole_number"),
            "preprocessing_group": rules.get("preprocessing_group"),
        }
    )
st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Validation report.
# ---------------------------------------------------------------------------
st.subheader("Semantic validation report")
validation_dir = RESULTS_DIR / "validation"
status_text = safe_read_text(validation_dir / "validation_status.txt")
summary_df = safe_read_csv(validation_dir / "validation_summary.csv")
issues_df = safe_read_csv(validation_dir / "validation_issues.csv")

if status_text is None:
    not_run_yet("Validation report", "validate")
else:
    st.info(status_text.strip())
    if summary_df is not None and not summary_df.empty:
        st.dataframe(summary_df, width="stretch", hide_index=True)
    if issues_df is not None and not issues_df.empty:
        st.markdown("**Issues**")
        st.dataframe(issues_df, width="stretch", hide_index=True)
    else:
        st.caption("No validation issues recorded.")
