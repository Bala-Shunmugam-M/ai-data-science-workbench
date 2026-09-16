"""
app.pages.11_New_Project
========================

PURPOSE
-------
Bring-your-own-dataset flow. Upload any CSV → the app profiles it, auto-detects
the task and target (you can override), then registers it as a project and runs
the full pipeline (prep → train → evaluate → auto-report). When it finishes the
new project is switchable in the sidebar like Housing and Churn.
"""

from __future__ import annotations

import os
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

from app.common import PAGE_ICON, page_header  # noqa: E402
from src.automl import detect as autodetect  # noqa: E402
from src.automl.register import register_project, slugify  # noqa: E402

st.set_page_config(page_title="New Project", page_icon="➕", layout="wide")
project_state.render_selector(_PROJECT)
page_header("➕ New Project", "Upload any CSV — the app profiles it, detects the task, trains, and registers it.")

upload = st.file_uploader("Dataset (CSV)", type=["csv"])
if upload is None:
    st.info("Upload a CSV to begin. The app auto-detects whether it's a regression "
            "or binary-classification problem, guesses the target column, and lets "
            "you confirm before training.")
    st.stop()


@st.cache_data(show_spinner="Reading dataset...")
def _read(file_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(file_bytes))


df = _read(upload.getvalue())
st.success(f"Loaded **{upload.name}** — {len(df):,} rows × {df.shape[1]} columns.")

with st.expander("Preview (first 20 rows)", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# --- Auto-detect + confirmation -------------------------------------------
st.subheader("1. Confirm the setup")
default_det = autodetect.detect(df)

col1, col2, col3 = st.columns([2, 2, 3])
target = col1.selectbox(
    "Target column", list(df.columns),
    index=list(df.columns).index(default_det.target),
)
detection = autodetect.detect(df, target=target)  # re-detect for the chosen target

task_choice = col2.selectbox(
    "Task", ["Auto-detected", "Regression", "Classification"],
    index=0,
)
display_name = col3.text_input("Project name", value=Path(upload.name).stem.replace("_", " ").title())

# Apply a task override if the user forces one.
if task_choice == "Regression":
    detection.task = "regression"
    detection.positive_class = None
    detection.n_classes = None
    detection.selection_metric = "rmse"
elif task_choice == "Classification":
    if detection.task != "classification":
        detection.task = "classification"
        detection.selection_metric = "roc_auc"
        detection.n_classes = int(df[target].nunique(dropna=True))
        detection.positive_class = (
            autodetect.guess_positive_class(df[target]) if detection.n_classes == 2 else None
        )

# Column profile.
with st.expander("Column profile", expanded=False):
    st.dataframe(pd.DataFrame(autodetect.profile_columns(df)), use_container_width=True, hide_index=True)

if detection.task == "classification":
    task_label = f"{'multiclass' if detection.is_multiclass else 'binary'} classification"
else:
    task_label = "regression"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Task", task_label)
c2.metric("Target", target)
c3.metric(
    "Classes" if detection.task == "classification" else "Positive class",
    str(detection.n_classes) if detection.task == "classification" else "—",
)
c4.metric("Predictors", df.shape[1] - 1 - len(detection.drop_columns))
if detection.positive_class:
    st.caption(f"Positive class: `{detection.positive_class}` (scored for ROC-AUC / precision / recall).")

for w in detection.warnings:
    st.warning(w)

if detection.task == "classification" and (detection.n_classes or 0) < 2:
    st.error(
        f"'{target}' has only one distinct value — there is nothing to learn. "
        "Pick a different target column."
    )
    st.stop()

# --- Register + run --------------------------------------------------------
st.subheader("2. Analyze & train")
st.caption(
    f"Registers the dataset as project `{slugify(display_name)}` and runs "
    "`autorun` (prep → train → evaluate → auto-report) in a subprocess."
)

if st.button("🚀 Analyze & Train", type="primary"):
    info = register_project(display_name, df, detection)
    slug = info["slug"]
    st.session_state["last_project"] = slug
    command = [sys.executable, "main.py", "autorun"]
    st.code(f"WORKBENCH_PROJECT={slug} " + " ".join(command), language="bash")

    env = {**os.environ, "WORKBENCH_PROJECT": slug}
    box = st.empty()
    log: list[str] = []
    proc = subprocess.Popen(
        command, cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line.rstrip())
        box.code("\n".join(log[-150:]), language="text")
    code = proc.wait()
    if code == 0:
        st.success(f"Done. Project **{display_name}** is ready and now appears in the sidebar switcher.")
    else:
        st.error(f"autorun failed (exit {code}). See the log above.")

# --- Results (if the last run for this upload finished) --------------------
slug = st.session_state.get("last_project")
if slug:
    ws = PROJECT_ROOT / "workspaces" / slug
    import json

    sel_path = ws / "artifacts" / "final_model_selection.json"
    report_path = ws / "artifacts" / "reports" / "auto_report.md"
    if sel_path.exists():
        st.subheader("3. Results")
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        items = project_state.champion_metric_items(sel)
        cols = st.columns(1 + len(items))
        cols[0].metric("Champion", f"{sel.get('champion_name')} {sel.get('champion_version')}")
        for col, (label, value) in zip(cols[1:], items):
            col.metric(label, value)

        figs_dir = ws / "artifacts" / "auto_eda"
        eval_figs = ws / "artifacts" / "evaluation" / "figures"
        imgs = list(figs_dir.glob("*.png")) + list(eval_figs.glob("*.png"))
        if imgs:
            gallery = st.columns(min(3, len(imgs)))
            for i, img in enumerate(imgs):
                gallery[i % len(gallery)].image(str(img), caption=img.stem.replace("_", " "), use_container_width=True)

        if report_path.exists():
            with st.expander("Automated report", expanded=True):
                st.markdown(report_path.read_text(encoding="utf-8"))

        if st.button(f"Open '{project_state.project_label(slug)}' in Dashboard"):
            st.query_params["project"] = slug
            st.switch_page("pages/1_Dashboard.py")
