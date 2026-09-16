"""
src.automl.report
=================

PURPOSE
-------
Generate the auto-EDA + data-quality summary and a self-contained report for an
uploaded dataset, using only the active project's descriptor (no hand-written
schema). Produces:

    artifacts/auto_eda/target_distribution.png
    artifacts/auto_eda/top_correlations.png
    artifacts/reports/auto_report.md    (also returned for live GUI rendering)

Called by the ``autorun`` command after evaluation, so it can fold the champion's
test metrics into the report.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import active  # noqa: E402
from config.paths import (  # noqa: E402
    ARTIFACTS_DIR,
    ENGINEERED_TRAIN_PATH,
    FINAL_MODEL_SELECTION_PATH,
    REPORTS_DIR,
    ensure_dir,
)
from src.model_training.design_matrix import encode_target  # noqa: E402
from src.utils.file_utils import load_csv, load_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

AUTO_EDA_DIR = ARTIFACTS_DIR / "auto_eda"
AUTO_REPORT_PATH = REPORTS_DIR / "auto_report.md"


def _target_series(df: pd.DataFrame, target: str, task: str, positive_class: str | None) -> pd.Series:
    if task == "classification":
        return encode_target(df[target], positive_class)
    return pd.to_numeric(df[target], errors="coerce")


def _is_multiclass_target(y: pd.Series) -> bool:
    """True when the (unencoded) classification target has 3+ classes."""

    return int(y.nunique(dropna=True)) > 2


def compute_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the auto-EDA summary dict (used by the report and the GUI)."""

    d = active.active_dataset()
    target, task = d.target, d.task
    predictors = [c for c in df.columns if c != target and c not in set(d.drop_columns)]

    missing = (
        df[predictors].isna().mean().mul(100).round(2).sort_values(ascending=False)
    )
    y = _target_series(df, target, task, d.positive_class)

    # Association of numeric predictors with the target. Pearson correlation is
    # only meaningful for a numeric/binary target; for a multiclass (nominal)
    # target we rank by ANOVA F instead, which is defined for 3+ classes.
    numeric_predictors = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    multiclass = task == "classification" and _is_multiclass_target(df[target])
    association_kind = "anova_f" if multiclass else "pearson"
    scores: dict[str, float] = {}

    if multiclass:
        from sklearn.feature_selection import f_classif

        usable = [c for c in numeric_predictors if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if usable:
            matrix = df[usable].apply(pd.to_numeric, errors="coerce")
            matrix = matrix.fillna(matrix.median(numeric_only=True))
            keep = [c for c in usable if matrix[c].std(skipna=True)]
            if keep:
                f_values, _ = f_classif(matrix[keep], df[target].astype("string"))
                scores = {
                    c: float(v) for c, v in zip(keep, f_values) if v == v  # drop NaN
                }
    else:
        for col in numeric_predictors:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.std(skipna=True) and y.std(skipna=True):
                scores[col] = float(np.corrcoef(s.fillna(s.median()), y.fillna(y.median()))[0, 1])

    top_corr = dict(sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])

    if task == "classification":
        counts = df[target].value_counts()
        target_summary = {
            "kind": "classes",
            "classes": {str(k): int(v) for k, v in counts.items()},
            "positive_class": d.positive_class,
            # A positive rate only exists for a binary target.
            "positive_rate": None if multiclass else round(float((y == 1).mean()), 4),
        }
    else:
        desc = pd.to_numeric(df[target], errors="coerce").describe()
        target_summary = {"kind": "numeric", "summary": {k: float(v) for k, v in desc.items()}}

    return {
        "n_rows": int(len(df)),
        "n_predictors": len(predictors),
        "n_numeric": len(numeric_predictors),
        "n_categorical": len(d.categorical_columns),
        "missing_top": missing[missing > 0].head(10).to_dict(),
        "top_correlations": top_corr,
        "association_kind": association_kind,
        "target_summary": target_summary,
    }


def _figures(df: pd.DataFrame, eda: dict[str, Any]) -> dict[str, str]:
    ensure_dir(AUTO_EDA_DIR)
    d = active.active_dataset()
    paths: dict[str, str] = {}

    # Target distribution.
    fig, ax = plt.subplots(figsize=(6, 4))
    ts = eda["target_summary"]
    if ts["kind"] == "classes":
        classes = ts["classes"]
        ax.bar(list(map(str, classes)), list(classes.values()), color="#4C78A8")
        ax.set_ylabel("Count")
        ax.set_title(f"Target distribution — {d.target}")
    else:
        ax.hist(pd.to_numeric(df[d.target], errors="coerce").dropna(), bins=40, color="#4C78A8", edgecolor="white")
        ax.set_xlabel(d.target)
        ax.set_ylabel("Count")
        ax.set_title(f"Target distribution — {d.target}")
    fig.tight_layout()
    p = AUTO_EDA_DIR / "target_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["target_distribution"] = str(p)

    # Top predictor associations (Pearson r, or ANOVA F for a multiclass target).
    if eda["top_correlations"]:
        anova = eda.get("association_kind") == "anova_f"
        items = list(eda["top_correlations"].items())
        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(items))))
        names = [k for k, _ in items][::-1]
        vals = [v for _, v in items][::-1]
        ax.barh(names, vals, color=["#E45756" if v < 0 else "#54A24B" for v in vals])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ANOVA F (class separation)" if anova else "Correlation with target")
        ax.set_title("Top predictors by class separation" if anova else "Top predictor correlations")
        fig.tight_layout()
        pc = AUTO_EDA_DIR / "top_correlations.png"
        fig.savefig(pc, dpi=120)
        plt.close(fig)
        paths["top_correlations"] = str(pc)
    return paths


def run() -> dict[str, Any]:
    """Build the auto-EDA figures + markdown report; return a summary dict."""

    d = active.active_dataset()
    df = load_csv(ENGINEERED_TRAIN_PATH)
    eda = compute_eda(df)
    figures = _figures(df, eda)

    selection = load_json(FINAL_MODEL_SELECTION_PATH) if FINAL_MODEL_SELECTION_PATH.exists() else None

    lines: list[str] = [
        f"# {d.display_name} — automated analysis",
        "",
        f"- **Task:** {d.task}  |  **Target:** `{d.target}`"
        + (f"  |  **Positive class:** `{d.positive_class}`" if d.positive_class else ""),
        f"- **Rows (train split):** {eda['n_rows']:,}  |  **Predictors:** {eda['n_predictors']} "
        f"({eda['n_numeric']} numeric, {eda['n_categorical']} categorical)",
        f"- **Selection metric:** {d.selection_metric}",
        "",
        "## Champion model",
    ]
    if selection:
        tm = selection.get("test_metrics", {})
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in tm.items())
        lines.append(
            f"**{selection.get('champion_name')} {selection.get('champion_version')}** — "
            f"test {metric_str}."
        )
    else:
        lines.append("_Not evaluated yet._")

    lines += ["", "## Target", ""]
    ts = eda["target_summary"]
    if ts["kind"] == "classes":
        balance = f"Class balance: {ts['classes']}"
        if ts.get("positive_rate") is not None:
            balance += f" (positive rate {ts['positive_rate']:.1%})"
        lines.append(balance + ".")
    else:
        s = ts["summary"]
        lines.append(f"mean={s.get('mean', 0):.3g}, std={s.get('std', 0):.3g}, "
                     f"min={s.get('min', 0):.3g}, max={s.get('max', 0):.3g}.")

    if eda["missing_top"]:
        lines += ["", "## Data quality — columns with missing values", ""]
        for col, pct in eda["missing_top"].items():
            lines.append(f"- `{col}`: {pct:.1f}% missing")

    if eda["top_correlations"]:
        if eda.get("association_kind") == "anova_f":
            lines += ["", "## Top predictors by class separation (ANOVA F)", ""]
            for col, c in eda["top_correlations"].items():
                lines.append(f"- `{col}`: F={c:,.1f}")
        else:
            lines += ["", "## Top predictor correlations with the target", ""]
            for col, c in eda["top_correlations"].items():
                lines.append(f"- `{col}`: {c:+.3f}")

    ensure_dir(REPORTS_DIR)
    AUTO_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Auto report written to %s", AUTO_REPORT_PATH)

    return {"eda": eda, "figures": figures, "report_path": str(AUTO_REPORT_PATH)}
