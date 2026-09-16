"""
app.pages.4_Features
=====================

PURPOSE
-------
The feature engineering plan (all five groups: Deterministic, Domain,
Statistical, Machine Learning, Inferential) joined with its governance
approval status.
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

from app.common import PAGE_ICON, not_run_yet, page_header, safe_read_json  # noqa: E402
from config.paths import FEATURE_APPROVAL_PATH, FEATURE_PROPOSAL_PATH  # noqa: E402

st.set_page_config(page_title="Features - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Features", "Governed feature engineering plan and its approval status.")

proposal = safe_read_json(FEATURE_PROPOSAL_PATH)
if proposal is None:
    not_run_yet("Feature proposal", "features")
    st.stop()

approval = safe_read_json(FEATURE_APPROVAL_PATH) or {}
approved_names = set(approval.get("approved_features", []))

c1, c2, c3 = st.columns(3)
c1.metric("Total proposed", proposal.get("total_proposed", 0))
c2.metric("Automatic (executed)", proposal.get("automatic_count", 0))
c3.metric("Approved", approval.get("approved_count", "n/a"))

features = proposal.get("features", [])
rows = []
for feature in features:
    name = feature.get("feature_name")
    rows.append(
        {
            "feature_id": feature.get("feature_id"),
            "feature_name": name,
            "group": feature.get("feature_group"),
            "subtype": feature.get("feature_subtype"),
            "formula": feature.get("formula"),
            "business_meaning": feature.get("business_meaning"),
            "automatic": feature.get("automatic"),
            "leakage_risk": feature.get("leakage_risk"),
            "priority": feature.get("priority"),
            "analyst_decision": feature.get("analyst_decision"),
            "approved": "✅ approved" if name in approved_names else "✗ not approved",
        }
    )
plan_df = pd.DataFrame(rows)

st.subheader("Feature plan")
group_filter = st.multiselect(
    "Filter by group", sorted(plan_df["group"].unique()) if not plan_df.empty else []
)
view = plan_df[plan_df["group"].isin(group_filter)] if group_filter else plan_df
st.dataframe(view, width="stretch", hide_index=True)

if approval:
    st.subheader("Approved predictor columns (feed the modelling matrix)")
    st.code(", ".join(approval.get("approved_predictor_columns", [])) or "none", language="text")
else:
    not_run_yet("Feature approval", "approve --features")
