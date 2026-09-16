"""
src.automl.register
===================

PURPOSE
-------
Turn an uploaded CSV + a confirmed :class:`~src.automl.detect.DatasetDetection`
into a registered, runnable project: it creates ``workspaces/<slug>/``, saves the
raw CSV there, and writes ``workspaces/<slug>/dataset.json`` (the descriptor the
dataset registry and ``config.active`` read). After this, the project is
switchable in the GUI and runnable with
``WORKBENCH_PROJECT=<slug> python main.py autorun``.

Paths are computed directly from ``PROJECT_ROOT`` (not ``config.paths``) because
registration runs while a *different* project may be active.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import PROJECT_ROOT
from src.automl.detect import DatasetDetection

WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
_RESERVED = {"california-housing", "california_housing", "churn"}


def slugify(name: str) -> str:
    """Filesystem/URL-safe slug from a display name; never collides with built-ins."""

    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or "dataset"
    slug = f"proj-{slug}" if slug in _RESERVED else slug
    # De-duplicate against existing workspaces.
    base, i = slug, 2
    while (WORKSPACES_DIR / slug / "dataset.json").exists():
        slug = f"{base}-{i}"
        i += 1
    return slug


def register_project(
    display_name: str,
    df: pd.DataFrame,
    detection: DatasetDetection,
    *,
    slug: str | None = None,
    file_name: str = "data.csv",
) -> dict[str, Any]:
    """
    Persist an uploaded dataset as a new project and return its descriptor dict.

    ``df`` is written verbatim as the project's raw data; ``detection`` supplies
    task/target/categoricals/etc. (typically after user confirmation).
    """

    if detection.task not in ("regression", "classification"):
        raise ValueError(
            f"Unsupported task '{detection.task}'. Only regression and "
            "classification can be registered."
        )

    # `detect()` already warns that a single-class target has nothing to learn,
    # but a warning stops nothing: registration succeeded, prep succeeded, and
    # the run died inside sklearn's `.fit()` with an error naming neither the
    # column nor the cause. Refuse here, where the message can be useful.
    if detection.task == "classification" and detection.n_classes == 1:
        raise ValueError(
            f"'{detection.target}' has only one distinct value, so there is "
            "nothing to learn. Pick a different target column."
        )

    if detection.target not in df.columns:
        raise ValueError(
            f"Target '{detection.target}' is not a column in this dataset. "
            f"Available columns: {', '.join(map(str, df.columns))}."
        )

    if df[detection.target].notna().sum() == 0:
        raise ValueError(
            f"Target '{detection.target}' is empty - every row is missing. "
            "Pick a different target column."
        )

    slug = slug or slugify(display_name)
    workspace = WORKSPACES_DIR / slug
    raw_dir = workspace / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / file_name
    df.to_csv(raw_path, index=False)

    descriptor: dict[str, Any] = {
        "name": slug,
        "display_name": display_name,
        "source": "local",
        "kaggle_id": "",
        "file": file_name,
        "task": detection.task,
        "target": detection.target,
        "positive_class": detection.positive_class,
        "stratify": "target" if detection.task == "classification" else "none",
        "selection_metric": detection.selection_metric,
        "categorical_columns": list(detection.categorical_columns),
        "drop_columns": list(detection.drop_columns),
        "description": f"Uploaded dataset ({len(df):,} rows).",
    }
    (workspace / "dataset.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    return {"slug": slug, "workspace": str(workspace), "descriptor": descriptor}
