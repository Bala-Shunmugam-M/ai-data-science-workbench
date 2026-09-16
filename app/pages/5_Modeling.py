"""
app.pages.5_Modeling
=====================

PURPOSE
-------
The governed model proposal/catalog, the approval status, per-model
parameters and metrics from the model registry, and the append-only
experiment log.
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
from config.paths import (  # noqa: E402
    EXPERIMENTS_LOG_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_PROPOSAL_PATH,
    MODEL_REGISTRY_PATH,
)

st.set_page_config(page_title="Modeling - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Modeling", "Model proposal, approvals, registry, and experiment log.")

proposal = safe_read_json(MODEL_PROPOSAL_PATH)
approval = safe_read_json(MODEL_APPROVAL_PATH)
registry = safe_read_json(MODEL_REGISTRY_PATH)

# ---------------------------------------------------------------------------
# Proposal / candidate catalog.
# ---------------------------------------------------------------------------
st.subheader("Model proposal")
if proposal is None:
    not_run_yet("Model proposal", "propose")
else:
    st.markdown(f"**Modelling scope:** {proposal.get('modelling_scope', 'n/a')}")
    st.markdown(f"**Rationale:** {proposal.get('rationale', 'n/a')}")
    st.markdown(f"**Recommended models:** {', '.join(proposal.get('recommended_models', []))}")
    with st.expander("Validation strategy"):
        for key, value in proposal.get("validation_strategy", {}).items():
            st.markdown(f"- **{key}**: {value}")
    catalog = proposal.get("candidate_catalog", [])
    if catalog:
        st.markdown("**Candidate catalog**")
        st.dataframe(pd.DataFrame(catalog), width="stretch", hide_index=True)
    flags = proposal.get("multicollinearity_flags", [])
    if flags:
        st.markdown("**Multicollinearity flags**")
        st.dataframe(pd.DataFrame(flags), width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Approval status.
# ---------------------------------------------------------------------------
st.subheader("Model approval")
if approval is None:
    not_run_yet("Model approval", "approve --models")
else:
    approved = set(approval.get("approved_models", []))
    recommended = proposal.get("recommended_models", []) if proposal else approved
    rows = [
        {"model": name, "approved": "✅ approved" if name in approved else "✗ not approved"}
        for name in (recommended or approved)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Model registry: every trained version, its params, and metrics.
# ---------------------------------------------------------------------------
st.subheader("Model registry")
if registry is None or not registry.get("models"):
    not_run_yet("Trained models", "train")
else:
    champion = registry.get("champion") or {}
    rows = []
    for model in registry["models"]:
        val = model.get("validation_metrics", {})
        rows.append(
            {
                "name": model.get("name"),
                "version": model.get("version"),
                "is_champion": model.get("name") == champion.get("name")
                and model.get("version") == champion.get("version"),
                "params": model.get("params"),
                "val_rmse": val.get("rmse"),
                "val_mae": val.get("mae"),
                "val_r2": val.get("r2"),
                "registered_at": model.get("registered_at"),
            }
        )
    registry_df = pd.DataFrame(rows).sort_values("val_rmse")
    st.dataframe(registry_df, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Experiment log (append-only history of every training run).
# ---------------------------------------------------------------------------
st.subheader("Experiment log")
experiments = safe_read_jsonl(EXPERIMENTS_LOG_PATH)
if not experiments:
    not_run_yet("Experiment log", "train")
else:
    st.dataframe(pd.DataFrame(experiments), width="stretch", hide_index=True)
