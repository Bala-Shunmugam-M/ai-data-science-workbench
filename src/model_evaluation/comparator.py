"""
src.model_evaluation.comparator
===============================

PURPOSE
-------
Build the model comparison table from per-model validation metrics and write a
plain-English model-selection narrative that a non-technical reader can follow.
Both are task-aware: regression ranks by RMSE (lower is better), classification
by ROC-AUC (higher is better), driven by the active project's selection metric.

PIPELINE POSITION
-----------------
Used by the evaluator to assemble ``model_comparison.csv`` and the selection
rationale embedded in ``final_model_selection.json`` and the reports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import active
from src.model_training import metrics as metric_helpers


def comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Return a comparison DataFrame sorted by the active selection metric.

    Each input row carries ``name``, ``version`` and a ``validation_metrics``
    dict. Every metric present is emitted as a ``validation_<metric>`` column,
    and a ``rank`` column (1 = best) is added using the metric's direction.
    """

    metric = active.selection_metric()
    flat: list[dict[str, Any]] = []
    for row in rows:
        vm = row.get("validation_metrics", {})
        record = {"model": row.get("name"), "version": row.get("version")}
        for key, value in vm.items():
            record[f"validation_{key}"] = value
        flat.append(record)

    table = pd.DataFrame(flat)
    if not table.empty:
        sort_col = f"validation_{metric}"
        higher = metric_helpers.METRIC_HIGHER_IS_BETTER.get(metric, False)
        if sort_col in table.columns:
            table = table.sort_values(sort_col, ascending=not higher).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
    return table


def _fmt_metrics(metrics: dict[str, float], task: str) -> str:
    """One-line metric summary appropriate to the task."""

    if task == "classification":
        parts = []
        for key in ("roc_auc", "accuracy", "precision", "recall", "f1"):
            if key in metrics:
                parts.append(f"{key.replace('_', '-').upper()} {metrics[key]:.3f}")
        return ", ".join(parts)
    return (
        f"RMSE {metrics.get('rmse', float('nan')):,.0f}, "
        f"MAE {metrics.get('mae', float('nan')):,.0f}, "
        f"R2 {metrics.get('r2', float('nan')):.3f}"
    )


def selection_narrative(
    table: pd.DataFrame,
    champion: dict[str, Any],
    champion_test_metrics: dict[str, float],
) -> str:
    """Compose a manager-readable explanation of why the champion was selected."""

    if table.empty:
        return "No models were available for comparison."

    task = active.task()
    metric = active.selection_metric()
    higher = metric_helpers.METRIC_HIGHER_IS_BETTER.get(metric, False)
    metric_label = metric.replace("_", "-").upper()

    lines: list[str] = []
    lines.append(
        f"{len(table)} candidate model(s) were compared on the untouched "
        f"validation split using {metric_label} as the selection metric "
        f"({'higher' if higher else 'lower'} is better)."
    )
    for _, r in table.iterrows():
        vm = {
            k[len("validation_"):]: r[k]
            for k in table.columns
            if k.startswith("validation_") and pd.notna(r[k])
        }
        lines.append(f"  - {r['model']} ({r['version']}): {_fmt_metrics(vm, task)}.")

    lines.append(
        f"{champion['name']} ({champion['version']}) had the best validation "
        f"{metric_label} and was promoted to champion."
    )
    lines.append(
        "The champion was then evaluated once on the held-out test split: "
        f"{_fmt_metrics(champion_test_metrics, task)}. "
        "The test split was used only for this single final measurement, never "
        "for tuning or selection."
    )
    return "\n".join(lines)
