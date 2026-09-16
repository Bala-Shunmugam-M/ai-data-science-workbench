"""
app.pages.8_Governance
========================

PURPOSE
-------
The append-only audit trail (parsed JSONL), the data/model lineage graph, and
the standing governance decisions registry.
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

from app.common import PAGE_ICON, not_run_yet, page_header, safe_read_json, safe_read_jsonl  # noqa: E402
from config.paths import AUDIT_LOG_PATH, DECISIONS_PATH, LINEAGE_PATH  # noqa: E402

st.set_page_config(page_title="Governance - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Governance", "Audit trail, lineage, and standing decisions.")

# ---------------------------------------------------------------------------
# Audit trail.
# ---------------------------------------------------------------------------
st.subheader("Audit trail")
events = safe_read_jsonl(AUDIT_LOG_PATH)
if not events:
    not_run_yet("Audit trail", "propose (or any governed stage)")
else:
    audit_df = pd.DataFrame(
        [
            {
                "timestamp": e.get("timestamp"),
                "actor": e.get("actor"),
                "event_type": e.get("event_type"),
                "payload": e.get("payload"),
            }
            for e in events
        ]
    )
    event_types = st.multiselect("Filter by event type", sorted(audit_df["event_type"].unique()))
    view = audit_df[audit_df["event_type"].isin(event_types)] if event_types else audit_df
    st.caption(f"{len(view)} of {len(audit_df)} event(s).")
    st.dataframe(view.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Lineage graph.
# ---------------------------------------------------------------------------
st.subheader("Data / model lineage")
lineage = safe_read_json(LINEAGE_PATH)
nodes = lineage.get("nodes", []) if lineage else []
if not nodes:
    not_run_yet("Lineage graph", "train / evaluate / explain")
else:
    st.caption(f"{len(nodes)} lineage node(s): {' -> '.join(n.get('stage', '?') for n in nodes)}")
    for node in nodes:
        with st.expander(f"{node.get('stage')} - {node.get('timestamp', '')}"):
            st.markdown(f"**Script:** `{node.get('script', 'n/a')}`")
            st.markdown("**Inputs**")
            st.json(node.get("inputs", []))
            st.markdown("**Outputs**")
            st.json(node.get("outputs", []))
            if node.get("params"):
                st.markdown("**Params**")
                st.json(node.get("params"))

st.divider()

# ---------------------------------------------------------------------------
# Standing governance decisions.
# ---------------------------------------------------------------------------
st.subheader("Standing decisions")
decisions_doc = safe_read_json(DECISIONS_PATH)
decisions = list((decisions_doc or {}).get("decisions", {}).values()) if decisions_doc else []
if not decisions:
    not_run_yet("Governance decisions", "propose")
else:
    for decision in decisions:
        with st.expander(decision.get("decision", decision.get("key", "decision"))):
            st.markdown(f"**Rationale:** {decision.get('rationale', 'n/a')}")
            st.caption(f"Recorded: {decision.get('date', 'n/a')}")
