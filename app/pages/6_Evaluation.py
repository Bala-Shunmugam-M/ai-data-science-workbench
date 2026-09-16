"""
app.pages.6_Evaluation
========================

PURPOSE
-------
The validation-set model comparison table, the champion card with its
once-only test metrics, and the champion's residual diagnostic figures.
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

from app.common import (  # noqa: E402
    PAGE_ICON,
    money,
    not_run_yet,
    page_header,
    safe_read_csv,
    safe_read_json,
)
from config.paths import (  # noqa: E402
    EVALUATION_FIGURES_DIR,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_COMPARISON_CSV,
    MODEL_EVALUATION_REPORT_PATH,
)

st.set_page_config(page_title="Evaluation - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Evaluation", "Validation comparison, champion selection, and test-set diagnostics.")

comparison_df = safe_read_csv(MODEL_COMPARISON_CSV)
evaluation_report = safe_read_json(MODEL_EVALUATION_REPORT_PATH)
final_selection = safe_read_json(FINAL_MODEL_SELECTION_PATH)

# ---------------------------------------------------------------------------
# Validation comparison.
# ---------------------------------------------------------------------------
st.subheader("Model comparison (validation split)")
if comparison_df is None:
    not_run_yet("Model comparison", "evaluate")
else:
    st.dataframe(comparison_df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Champion card.
# ---------------------------------------------------------------------------
st.subheader("Champion")
if final_selection is None:
    not_run_yet("Final model selection", "evaluate")
else:
    items = project_state.champion_metric_items(final_selection)
    cols = st.columns(1 + len(items))
    cols[0].metric("Champion", f"{final_selection['champion_name']} {final_selection['champion_version']}")
    for col, (label, value) in zip(cols[1:], items):
        col.metric(label, value)
    st.caption(f"Selection metric: {final_selection.get('selection_metric', 'rmse')} (on the validation split)")

    with st.expander("Selection narrative", expanded=True):
        st.text(final_selection.get("selection_rationale", "n/a"))

st.divider()

# ---------------------------------------------------------------------------
# Champion diagnostics (from the single test-set evaluation): residuals for
# regression, confusion matrix + ROC for classification.
# ---------------------------------------------------------------------------
st.subheader("Champion diagnostics (test split)")
figures = (
    (evaluation_report.get("champion_figures") or evaluation_report.get("residual_figures") or {})
    if evaluation_report else {}
)
if not figures:
    not_run_yet("Residual figures", "evaluate")
else:
    columns = st.columns(len(figures))
    for column, (name, path) in zip(columns, figures.items()):
        path_obj = Path(path)
        if path_obj.exists():
            column.image(str(path_obj), caption=name.replace("_", " "), width="stretch")
        else:
            column.caption(f"{name}: figure file missing.")
