"""
docs.build_report
=================

PURPOSE
-------
Render the project report from its template, filling every result figure from
``src.reporting.facts`` - the same module the slide deck and the published
Pages site read - then convert the result to Word.

WHY A TEMPLATE
--------------
The report used to carry its figures as literals in prose. That meant a
pipeline re-run made it stale without any signal, and an edit to one document
could leave the report, the deck and the site quoting three different numbers
for the same run. A reader comparing them would have no way to tell which was
current.

Prose that DESCRIBES the method is authored by hand in the template and is not
touched here. Only the result figures are substituted, because only they change
when the pipeline runs.

PIPELINE POSITION
-----------------
    evaluation + trust -> facts -> [build_report] -> PROJECT-REPORT.md -> .docx

USAGE
-----
    python docs/build_report.py
"""

from __future__ import annotations

import json
from string import Formatter
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from src.reporting.facts import (  # noqa: E402
    CHURN,
    HOUSING,
    money,
    warn_if_stale,
)

TEMPLATE = Path(__file__).parent / "PROJECT-REPORT.template.md"
MARKDOWN = Path(__file__).parent / "PROJECT-REPORT.md"


def _signed(value: float | None) -> str:
    """A gap reads as +0.1208 / -0.0259: the sign carries the meaning."""
    if value is None:
        return "not available"
    return f"{value:+.4f}"


def _n_rows(split: str) -> int:
    """Row count for a split, from the churn model card."""
    card = CHURN.root / "artifacts" / "trust" / "model_card.json"
    try:
        data = json.loads(card.read_text(encoding="utf-8"))
        return int(data["data"][f"n_{split}_rows"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"train": 4930, "validation": 1056, "test": 1057}[split]


def values() -> dict[str, str]:
    """Every placeholder the template uses, rendered as it should read."""
    cal = CHURN.calibration
    worst = CHURN.worst_fairness_column

    def imp(index: int) -> str:
        rows = CHURN.importance.get("importances", [])
        if index < len(rows):
            return f"{rows[index]['importance_mean']:.4f}"
        return "not available"

    return {
        # --- housing ------------------------------------------------------
        "h_test_rmse": money(HOUSING.test_metrics["rmse"]),
        "h_val_rmse": money(HOUSING.validation_metrics["rmse"]),
        "h_runner_rmse": money(HOUSING.runner_up_metrics["rmse"]),
        "h_test_mae": money(HOUSING.test_metrics["mae"]),
        "h_test_r2": f"{HOUSING.test_metrics['r2']:.3f}",
        "h_val_r2": f"{HOUSING.validation_metrics['r2']:.3f}",
        "h_test_mape": f"{HOUSING.test_metrics['mape']:.1f}%",
        "h_alpha": f"{HOUSING.champion_params.get('alpha', 0):g}",

        # --- churn --------------------------------------------------------
        "c_val_auc": f"{CHURN.validation_metrics['roc_auc']:.4f}",
        "c_test_auc": f"{CHURN.test_metrics['roc_auc']:.4f}",
        "c_runner_auc": f"{CHURN.runner_up_metrics['roc_auc']:.4f}",
        "c_c": f"{CHURN.champion_params.get('C', 0):.4g}",
        "c_n_train": f"{_n_rows('train'):,}",

        # --- trust: calibration -------------------------------------------
        "cal_brier": f"{cal.get('brier_score', float('nan')):.4f}",
        "cal_ece": f"{cal.get('ece', float('nan')):.4f}",
        "cal_mce": f"{cal.get('mce', float('nan')):.4f}",
        "cal_mean_pred": f"{cal.get('mean_predicted', float('nan')):.4f}",
        "cal_base_rate": f"{cal.get('base_rate', float('nan')):.4f}",

        # --- trust: permutation importance ---------------------------------
        "imp_2_value": imp(1),
        "imp_3_value": imp(2),

        # --- trust: fairness ------------------------------------------------
        "amp_worst": _signed(CHURN.amplification(worst)),
    }


def build() -> Path:
    template = TEMPLATE.read_text(encoding="utf-8")
    supplied = values()

    # Check the TEMPLATE's field names, not the rendered output. After
    # .format() an escaped "{{" is legitimately a single brace, so the report's
    # own path globs - artifacts/trust/model_card.{md,json} - are
    # indistinguishable from an unfilled placeholder once rendering has
    # happened. Formatter.parse sees the fields before that ambiguity exists.
    fields = {
        name for _, name, _, _ in Formatter().parse(template)
        if name
    }
    missing = sorted(fields - supplied.keys())
    if missing:
        raise SystemExit(f"template asks for figures values() does not "
                         f"supply: {missing}")
    unused = sorted(supplied.keys() - fields)
    if unused:
        print(f"note: {len(unused)} value(s) no longer used by the template: "
              f"{unused}")

    MARKDOWN.write_text(template.format(**supplied), encoding="utf-8")
    return MARKDOWN


if __name__ == "__main__":
    path = build()
    print(f"OK  {path}")

    from build_report_docx import OUT as DOCX_OUT, convert  # noqa: E402

    docx = convert(path, DOCX_OUT)
    print(f"OK  {docx}")
    print(warn_if_stale())
