"""
app.pages.9_Reports
=====================

PURPOSE
-------
Preview and download the self-contained HTML project report, plus quick
download links for the other key managerial/governance deliverables.
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
import streamlit.components.v1 as components  # noqa: E402

from app.common import PAGE_ICON, not_run_yet, page_header, safe_read_text  # noqa: E402
from config.paths import (  # noqa: E402
    EXECUTIVE_BRIEFING_PATH,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_EVALUATION_REPORT_PATH,
    PROJECT_REPORT_PATH,
)

st.set_page_config(page_title="Reports - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Reports", "The self-contained HTML project report and other deliverables.")

# ---------------------------------------------------------------------------
# Project report.
# ---------------------------------------------------------------------------
st.subheader("Project report (self-contained HTML)")
report_html = safe_read_text(PROJECT_REPORT_PATH)
if report_html is None:
    not_run_yet("Project report", "report")
else:
    st.download_button(
        "Download project_report.html",
        data=report_html,
        file_name="project_report.html",
        mime="text/html",
    )
    st.caption(f"File: {PROJECT_REPORT_PATH}")
    with st.expander("Preview (rendered in an iframe)", expanded=True):
        components.html(report_html, height=900, scrolling=True)

st.divider()

# ---------------------------------------------------------------------------
# Other deliverables.
# ---------------------------------------------------------------------------
st.subheader("Other deliverables")

for path, label, mime in [
    (EXECUTIVE_BRIEFING_PATH, "Executive briefing (Markdown)", "text/markdown"),
    (FINAL_MODEL_SELECTION_PATH, "Final model selection (JSON)", "application/json"),
    (MODEL_EVALUATION_REPORT_PATH, "Model evaluation report (JSON)", "application/json"),
]:
    text = safe_read_text(path)
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**{label}** - `{path}`")
    if text is None:
        col2.caption("not generated yet")
    else:
        col2.download_button("Download", data=text, file_name=path.name, mime=mime, key=str(path))
