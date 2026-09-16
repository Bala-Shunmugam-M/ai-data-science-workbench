"""
app.pages.1_Dashboard
======================

PURPOSE
-------
Project KPIs (rows, split sizes, feature/model counts, champion + test
metrics) and the orchestrator's pipeline stage-status table. The only
mutating control in the whole GUI lives here: "Run stage", which shells out
to ``python main.py pipeline ...`` and streams its output - it does not call
any pipeline function in-process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
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
    FEATURE_APPROVAL_PATH,
    FEATURE_PROPOSAL_PATH,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_PROPOSAL_PATH,
    MODEL_REGISTRY_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from src.workflow import orchestrator, pipeline_builder

st.set_page_config(page_title="Dashboard - AI Data Science Workbench", page_icon=PAGE_ICON, layout="wide")
project_state.render_selector(_PROJECT)
page_header("Dashboard", "Project KPIs and pipeline stage status.")

# ---------------------------------------------------------------------------
# KPI row 1: dataset size.
# ---------------------------------------------------------------------------
train_df = safe_read_csv(TRAIN_PATH)
val_df = safe_read_csv(VALIDATION_PATH)
test_df = safe_read_csv(TEST_PATH)

st.subheader("Dataset")
if train_df is None:
    not_run_yet("Splits", "preprocess")
else:
    total_rows = len(train_df) + (len(val_df) if val_df is not None else 0) + (
        len(test_df) if test_df is not None else 0
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{total_rows:,}")
    c2.metric("Train rows", f"{len(train_df):,}")
    c3.metric("Validation rows", f"{len(val_df):,}" if val_df is not None else "n/a")
    c4.metric("Test rows", f"{len(test_df):,}" if test_df is not None else "n/a")

# ---------------------------------------------------------------------------
# KPI row 2: features + models.
# ---------------------------------------------------------------------------
st.subheader("Features and models")
feature_proposal = safe_read_json(FEATURE_PROPOSAL_PATH)
feature_approval = safe_read_json(FEATURE_APPROVAL_PATH)
model_proposal = safe_read_json(MODEL_PROPOSAL_PATH)
model_approval = safe_read_json(MODEL_APPROVAL_PATH)
registry = safe_read_json(MODEL_REGISTRY_PATH)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Features proposed",
    feature_proposal.get("total_proposed", "n/a") if feature_proposal else "n/a",
)
c2.metric(
    "Features approved",
    feature_approval.get("approved_count", "n/a") if feature_approval else "n/a",
)
c3.metric(
    "Models recommended",
    len(model_proposal.get("recommended_models", [])) if model_proposal else "n/a",
)
c4.metric(
    "Models trained",
    len(registry.get("models", [])) if registry else 0,
)
if model_approval:
    st.caption(f"Approved models: {', '.join(model_approval.get('approved_models', [])) or 'none'}")

# ---------------------------------------------------------------------------
# Champion card.
# ---------------------------------------------------------------------------
st.subheader("Champion model")
final_selection = safe_read_json(FINAL_MODEL_SELECTION_PATH)
if final_selection is None:
    not_run_yet("Champion selection", "evaluate")
else:
    items = project_state.champion_metric_items(final_selection)
    cols = st.columns(1 + len(items))
    cols[0].metric("Champion", f"{final_selection['champion_name']} {final_selection['champion_version']}")
    for col, (label, value) in zip(cols[1:], items):
        col.metric(label, value)

st.divider()

# ---------------------------------------------------------------------------
# Pipeline stage status (from artifacts/workflow_status.json).
# ---------------------------------------------------------------------------
st.subheader("Pipeline stage status")
status = orchestrator.get_status()
stage_names = pipeline_builder.STAGE_NAMES
rows = []
for name in stage_names:
    entry = status.get(name, {})
    rows.append(
        {
            "stage": name,
            "status": entry.get("status", "not run"),
            "last_run": entry.get("last_run") or "",
            "duration_s": float(entry["duration_s"]) if "duration_s" in entry else float("nan"),
            "n_outputs": len(entry.get("outputs", [])) if entry.get("outputs") else 0,
        }
    )
status_df = pd.DataFrame(rows)


def _highlight(row: pd.Series) -> list[str]:
    color = {
        "ok": "background-color: #1e5b3a",
        "skipped": "background-color: #4a4a20",
        "failed": "background-color: #6b1f1f",
        "not run": "",
    }.get(row["status"], "")
    return [color] * len(row)


st.dataframe(status_df.style.apply(_highlight, axis=1), width="stretch", hide_index=True)
if not status:
    st.caption(
        "No stage has been run through the orchestrator yet. Individual "
        "`python main.py <stage>` runs are not reflected here until the "
        "stage is also run via `python main.py pipeline` or `all`."
    )

st.divider()

# ---------------------------------------------------------------------------
# Run stage - the ONE mutating control in the whole GUI. Shells out to the
# CLI (never calls pipeline code in-process) and streams stdout/stderr.
# ---------------------------------------------------------------------------
st.subheader("Run stage")


def _stream_command(command: list[str]) -> int:
    """Shell out with the active project in the environment; stream output live."""

    import os

    st.code(" ".join(command), language="bash")
    env = {**os.environ, "WORKBENCH_PROJECT": _PROJECT}
    output_box = st.empty()
    log_lines: list[str] = []
    process = subprocess.Popen(
        command, cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line.rstrip())
        output_box.code("\n".join(log_lines[-200:]), language="text")
    return process.wait()


if _PROJECT == "churn":
    # The housing feature/EDA stages are schema-bound, so churn runs its own
    # one-shot prep -> train -> evaluate command.
    st.caption(
        "Shells out to `WORKBENCH_PROJECT=churn python main.py churn-all` "
        "(prep -> train -> evaluate) and streams its output. The only control "
        "in the GUI that changes project state."
    )
    if st.button("Run churn pipeline (prep → train → evaluate)"):
        code = _stream_command([sys.executable, "main.py", "churn-all"])
        (st.success if code == 0 else st.error)(
            "Churn pipeline completed." if code == 0 else f"Churn pipeline failed (exit {code})."
        )
        st.rerun()
else:
    st.caption(
        "Shells out to `python main.py pipeline --from ... --to ... [--force]` "
        "and streams its output below. This is the only control in the GUI that "
        "changes project state."
    )
    with st.form("run_stage_form"):
        col1, col2, col3 = st.columns(3)
        from_stage = col1.selectbox("From", stage_names, index=0)
        to_stage = col2.selectbox("To", stage_names, index=len(stage_names) - 1)
        force = col3.checkbox("Force rerun", value=False)
        submitted = st.form_submit_button("Run pipeline range")

    if submitted:
        if stage_names.index(from_stage) > stage_names.index(to_stage):
            st.error(f"'{from_stage}' comes after '{to_stage}' in the stage order.")
        else:
            command = [sys.executable, "main.py", "pipeline", "--from", from_stage, "--to", to_stage]
            if force:
                command.append("--force")
            code = _stream_command(command)
            if code == 0:
                st.success(f"Pipeline range '{from_stage}' -> '{to_stage}' completed.")
            else:
                st.error(f"Pipeline range failed (exit code {code}). See output above.")
            st.rerun()
