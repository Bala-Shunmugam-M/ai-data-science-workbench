"""
app.common
==========

PURPOSE
-------
Shared, READ-ONLY helpers for every Streamlit page: safe CSV/JSON/text
loaders that degrade to ``None`` instead of raising (so a page can render
"stage not run yet" instead of crashing), a few small formatting helpers, and
one page-header helper. No function in this module writes to disk.

PIPELINE POSITION
-----------------
Imported by ``app/main.py`` and every ``app/pages/*.py`` script. Each caller
is responsible for putting ``PROJECT_ROOT`` on ``sys.path`` before importing
this module (see the top of any page for the two-line snippet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PAGE_ICON = "\U0001F4CA"  # bar chart emoji


# ---------------------------------------------------------------------------
# Safe loaders - never raise, always return None (or an empty frame) when the
# artifact has not been generated yet.
# ---------------------------------------------------------------------------
def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    """Return a DataFrame, or ``None`` when ``path`` is missing/unreadable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:  # noqa: BLE001 - a read-only dashboard must never crash.
        return None


def safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    """Return parsed JSON, or ``None`` when ``path`` is missing/unreadable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return every record of a JSONL file, or ``[]`` when missing/unreadable."""

    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception:  # noqa: BLE001
        return records
    return records


def safe_read_text(path: Path) -> str | None:
    """Return file text, or ``None`` when ``path`` is missing/unreadable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# UI helpers.
# ---------------------------------------------------------------------------
def page_header(title: str, caption: str = "") -> None:
    """Render a consistent page title (+ optional caption)."""

    st.title(title)
    if caption:
        st.caption(caption)


def not_run_yet(label: str, command: str) -> None:
    """Render a consistent "stage not run yet" notice."""

    st.info(f"{label} not available yet - run `python main.py {command}`.")


def money(value: Any) -> str:
    """Format a number as a dollar amount, or ``"n/a"`` when it can't be."""

    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def pct(value: Any, digits: int = 1) -> str:
    """Format a fraction (0-1) or already-percent number as ``NN.N%``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(number) <= 1.0:
        number *= 100
    return f"{number:.{digits}f}%"


def number(value: Any, digits: int = 3) -> str:
    """Format a plain number, or ``"n/a"`` when it can't be."""

    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def approval_badge(is_approved: bool) -> str:
    """Small text badge for a boolean approval flag (no HTML needed)."""

    return "✅ approved" if is_approved else "✗ not approved"
