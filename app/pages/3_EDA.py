"""
app.pages.3_EDA
================

PURPOSE
-------
Data-understanding tables (structure/quality on the training split) plus the
EDA figure gallery and key analytical-summary tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import project_state  # noqa: E402

_PROJECT = project_state.apply_project()

from app.common import PAGE_ICON, not_run_yet, page_header, safe_read_csv, safe_read_text  # noqa: E402
from config.paths import RESULTS_DIR  # noqa: E402

st.set_page_config(page_title="EDA - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("EDA", "Data understanding + exploratory analysis on the training split.")

UNDERSTANDING_DIR = RESULTS_DIR / "data_understanding"
EDA_DIR = RESULTS_DIR / "eda"
FIGURES_DIR = EDA_DIR / "figures"
SUMMARY_DIR = RESULTS_DIR / "eda_analysis_summary"

# ---------------------------------------------------------------------------
# Data understanding.
# ---------------------------------------------------------------------------
st.subheader("Data understanding")
dataset_summary = safe_read_csv(UNDERSTANDING_DIR / "dataset_summary.csv")
if dataset_summary is None:
    not_run_yet("Data-understanding report", "understand")
else:
    st.dataframe(dataset_summary, width="stretch", hide_index=True)
    tabs = st.tabs(["Variables", "Missing values", "Categorical frequencies", "Logical checks"])
    for tab, filename in zip(
        tabs,
        ["variable_summary.csv", "missing_values.csv", "categorical_frequencies.csv", "logical_consistency_checks.csv"],
    ):
        with tab:
            df = safe_read_csv(UNDERSTANDING_DIR / filename)
            if df is None or df.empty:
                st.caption("No rows.")
            else:
                st.dataframe(df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# EDA analytical summary tables.
# ---------------------------------------------------------------------------
st.subheader("Analytical summary")
conclusion = safe_read_text(SUMMARY_DIR / "eda_conclusion.txt")
if conclusion is None:
    not_run_yet("EDA analytical summary", "eda")
else:
    with st.expander("EDA conclusion", expanded=True):
        st.text(conclusion)
    tabs = st.tabs(["Candidate predictors", "Modelling challenges", "Nonlinearity screen", "Correlation screen"])
    for tab, filename in zip(
        tabs,
        [
            "candidate_explanatory_variables.csv",
            "modelling_challenges.csv",
            "nonlinearity_screen.csv",
            "predictor_correlation_screen.csv",
        ],
    ):
        with tab:
            df = safe_read_csv(SUMMARY_DIR / filename)
            if df is None or df.empty:
                st.caption("No rows.")
            else:
                st.dataframe(df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Figure gallery.
# ---------------------------------------------------------------------------
st.subheader("EDA figure gallery")
if not FIGURES_DIR.exists():
    not_run_yet("EDA figures", "eda")
else:
    figures = sorted(FIGURES_DIR.glob("*.png"))
    if not figures:
        st.caption("No figures found.")
    else:
        query = st.text_input("Filter figures by filename", "")
        if query:
            figures = [f for f in figures if query.lower() in f.name.lower()]
        st.caption(f"{len(figures)} figure(s).")
        columns = st.columns(3)
        for index, figure_path in enumerate(figures):
            with columns[index % 3]:
                st.image(str(figure_path), caption=figure_path.stem.replace("_", " "), width="stretch")
