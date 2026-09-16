"""
app.pages.7_Explainability
============================

PURPOSE
-------
The champion's standardized-coefficient table, the top-drivers chart, and
the rendered managerial executive briefing.
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
from config.paths import EXECUTIVE_BRIEFING_PATH, EXPLAINABILITY_DIR  # noqa: E402

st.set_page_config(page_title="Explainability - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Explainability", "Standardized coefficients, top drivers, and the executive briefing.")

COEFFICIENT_CSV = EXPLAINABILITY_DIR / "coefficient_interpretation.csv"
TOP_DRIVERS_PNG = EXPLAINABILITY_DIR / "top_drivers.png"

# ---------------------------------------------------------------------------
# Coefficient table.
# ---------------------------------------------------------------------------
st.subheader("Standardized coefficients")
coefficients_df = safe_read_csv(COEFFICIENT_CSV)
if coefficients_df is None:
    not_run_yet("Coefficient interpretation", "explain")
else:
    st.dataframe(coefficients_df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Top-drivers chart.
# ---------------------------------------------------------------------------
st.subheader("Top drivers")
if TOP_DRIVERS_PNG.exists():
    st.image(str(TOP_DRIVERS_PNG), width="stretch")
else:
    not_run_yet("Top-drivers chart", "explain")

st.divider()

# ---------------------------------------------------------------------------
# Executive briefing (managerial, non-mathematical).
# ---------------------------------------------------------------------------
st.subheader("Executive briefing")
briefing = safe_read_text(EXECUTIVE_BRIEFING_PATH)
if briefing is None:
    not_run_yet("Executive briefing", "explain")
else:
    st.markdown(briefing)
    st.download_button(
        "Download executive_briefing.md",
        data=briefing,
        file_name="executive_briefing.md",
        mime="text/markdown",
    )
