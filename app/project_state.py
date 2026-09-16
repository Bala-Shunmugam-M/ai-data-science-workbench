"""
app.project_state
==================

PURPOSE
-------
Make the GUI a live, in-process **two-project** dashboard. Streamlit caches
imported modules, so switching the active project means re-resolving the
workspace paths: :func:`apply_project` sets ``WORKBENCH_PROJECT`` from the URL
query param and reloads :mod:`config.paths` + :mod:`config.active` so every
``from config.paths import ...`` executed *after* it sees the selected project.

Each page calls :func:`apply_project` at the top (before it imports path
constants) and :func:`render_selector` in the sidebar (after ``set_page_config``).

Also holds small task-aware helpers so the champion KPIs read RMSE/R² for the
regression showcase and ROC-AUC/accuracy for the churn classifier.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

#: Built-in projects, always listed first.
_BUILTIN: dict[str, str] = {
    "california-housing": "🏠 California Housing (regression)",
    "churn": "🎯 Customer Churn (classification)",
}
_DEFAULT = "california-housing"
_WORKSPACES = Path(__file__).resolve().parents[1] / "workspaces"


def available_projects() -> dict[str, str]:
    """slug -> label for every project: built-ins plus uploaded workspaces."""

    projects = dict(_BUILTIN)
    if _WORKSPACES.exists():
        for descriptor in sorted(_WORKSPACES.glob("*/dataset.json")):
            slug = descriptor.parent.name
            if slug in projects:
                continue
            try:
                spec = json.loads(descriptor.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task = spec.get("task", "")
            label = spec.get("display_name", slug)
            projects[slug] = f"📄 {label} ({task})"
    return projects


def project_label(slug: str) -> str:
    return available_projects().get(slug, slug)


def apply_project() -> str:
    """
    Resolve the selected project from the URL query param, activate it, and
    reload the path/config modules so subsequent imports see it. Returns the slug.
    """

    slug = st.query_params.get("project", os.environ.get("WORKBENCH_PROJECT", _DEFAULT))
    if slug not in available_projects():
        slug = _DEFAULT
    os.environ["WORKBENCH_PROJECT"] = slug

    import config.paths as paths
    import config.active as active

    importlib.reload(paths)
    importlib.reload(active)
    return slug


def render_selector(current: str) -> None:
    """Render the sidebar project switcher; change the URL + rerun on switch."""

    projects = available_projects()
    slugs = list(projects)
    st.sidebar.markdown("### Project")
    choice = st.sidebar.radio(
        "Active project",
        slugs,
        index=slugs.index(current) if current in slugs else 0,
        format_func=lambda s: projects[s],
        key="project_selector",
        label_visibility="collapsed",
    )
    if choice != current:
        st.query_params["project"] = choice
        st.rerun()
    st.sidebar.caption(f"Workspace: `{choice}`")
    st.sidebar.page_link("pages/11_New_Project.py", label="➕ New project (upload CSV)")


# ---------------------------------------------------------------------------
# Task-aware champion helpers (shared by Home, Dashboard, Evaluation).
# ---------------------------------------------------------------------------
def _fmt(metric: str, value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if metric in {"rmse", "mae"}:
        return f"${value:,.0f}"
    if metric == "mape":
        return f"{value:.1f}%"
    return f"{value:.3f}"


def champion_metric_items(final_selection: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (label, formatted-value) pairs appropriate to the champion's task."""

    task = final_selection.get("task", "regression")
    tm = final_selection.get("test_metrics") or {}
    if task == "classification":
        order = [("roc_auc", "Test ROC-AUC"), ("accuracy", "Test accuracy"),
                 ("precision", "Test precision"), ("recall", "Test recall")]
    else:
        # Fall back to the legacy flat keys the housing report also uses.
        tm = tm or {
            "rmse": final_selection.get("test_rmse"),
            "mae": final_selection.get("test_mae"),
            "r2": final_selection.get("test_r2"),
        }
        order = [("rmse", "Test RMSE"), ("mae", "Test MAE"), ("r2", "Test R²")]
    return [(label, _fmt(key, tm.get(key))) for key, label in order]


def champion_summary_line(final_selection: dict[str, Any]) -> str:
    """One-line champion summary for the Home banner."""

    name = f"{final_selection.get('champion_name')} {final_selection.get('champion_version')}"
    items = ", ".join(f"{label} {value}" for label, value in champion_metric_items(final_selection))
    return f"Champion model: **{name}** — {items}."
