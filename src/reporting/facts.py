"""
src.reporting.facts
===================

PURPOSE
-------
THE single source of every number that appears in a deliverable.

Three artefacts quote this project's results - the slide deck, the written
report, and the published Pages site. Each used to carry its own transcribed
copy of the figures, which meant a pipeline re-run silently made all three
stale, and any hand-edit could make them disagree with each other in front of
a reader. This module ends that: every deliverable imports from here, so they
cannot drift apart, and a re-run updates all of them at once.

DESIGN
------
Values are read from the pipeline's own artifacts at import time. Every lookup
carries the figure verified on 2026-09-08 as a fallback, and any fallback used
is recorded in :data:`STALE`. A deliverable generator therefore still builds
when an artifact is missing - printing a warning rather than failing - because
a generator that refuses to run the night before a defence is a worse failure
than one showing a slightly old number.

Formatting lives here too. If the deck rounds ROC-AUC to four places and the
report rounds to three, the two documents disagree on paper even when the
underlying value is identical, so the canonical strings are defined once.

PIPELINE POSITION
-----------------
    evaluation + trust -> [facts] -> deck / report / pages site

USAGE
-----
    from src.reporting.facts import HOUSING, CHURN, STALE

    print(HOUSING.champion_label)      # "Ridge v001"
    print(HOUSING.test_line)           # "RMSE $66,681  .  R2 0.667"
    print(CHURN.calibration["ece"])    # 0.030095

Self-check:
    python src/reporting/facts.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Every fallback below was read from the artifacts on this date. If a value is
# served from a fallback the reason is appended to STALE.
VERIFIED_ON = "2026-09-08"

STALE: list[str] = []

SEP = "  ·  "  # the separator between metrics, in every deliverable


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _get(data: dict | None, *keys: str, default: Any, where: str) -> Any:
    """Walk `keys` through `data`; fall back loudly if anything is absent."""
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            STALE.append(f"{where} :: {'.'.join(keys)}")
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Formatting - defined once so no two deliverables round differently
# ---------------------------------------------------------------------------
def money(value: float) -> str:
    """Currency-scale metric, e.g. 66680.81 -> '$66,681'."""
    return f"${value:,.0f}"


def metric(name: str, value: float) -> str:
    """
    One metric as it should read in every deliverable.

    RMSE and MAE are in target units, which for this project's regression case
    is dollars; everything else is a ratio or a percentage.
    """
    label = name.upper().replace("_", "-")
    if name in ("rmse", "mae") and abs(value) >= 1000:
        return f"{label} {money(value)}"
    if name == "mape":
        return f"{label} {value:.2f}%"
    if name == "r2":
        return f"R2 {value:.3f}"
    return f"{label} {value:.4f}"


def metric_line(metrics: dict[str, float] | None, *keys: str) -> str:
    """
    A headline row of metrics, e.g. 'RMSE $66,681  .  R2 0.667'.

    Passing `keys` selects and orders them; omitting it renders everything.
    An absent metric set renders as 'not available' rather than as an empty
    string, so a missing number is visible in the deliverable instead of
    looking like a value nobody thought to include.
    """
    if not metrics:
        return "not available"
    names = keys or tuple(metrics)
    parts = [
        metric(name, metrics[name])
        for name in names
        if isinstance(metrics.get(name), (int, float))
    ]
    return SEP.join(parts) or "not available"


# ---------------------------------------------------------------------------
# The facts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectFacts:
    """Everything a deliverable may say about one project."""

    slug: str
    display_name: str
    task: str
    root: Path

    champion_name: str
    champion_version: str
    champion_params: dict[str, Any]
    selection_metric: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    n_models: int
    runner_up_name: str
    runner_up_metrics: dict[str, float]

    headline_keys: tuple[str, ...]
    fairness: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    importance: dict[str, Any] = field(default_factory=dict)

    # -- derived, canonical strings ----------------------------------------
    @property
    def champion_label(self) -> str:
        return f"{self.champion_name} {self.champion_version}".strip()

    @property
    def validation_line(self) -> str:
        return metric_line(self.validation_metrics, *self.headline_keys)

    @property
    def test_line(self) -> str:
        return metric_line(self.test_metrics, *self.headline_keys)

    @property
    def runner_up_line(self) -> str:
        if not self.runner_up_name:
            return "not available"
        value = self.runner_up_metrics.get(self.selection_metric)
        if not isinstance(value, (int, float)):
            return self.runner_up_name
        return f"{self.runner_up_name}, {metric(self.selection_metric, value)}"

    @property
    def tuned_parameter(self) -> str:
        """The tuned hyperparameter as 'alpha = 100', or '' when untuned."""
        for key in ("alpha", "C", "max_depth"):
            if key in self.champion_params:
                # 4 significant figures: enough to identify the value that was
                # selected, without printing C = 2.154434690031882 on a slide.
                return f"{key} = {self.champion_params[key]:.4g}"
        return ""

    # -- trust conveniences -------------------------------------------------
    @property
    def top_features(self) -> list[tuple[str, float]]:
        rows = self.importance.get("importances", [])
        return [(r["feature"], r["importance_mean"]) for r in rows[:3]]

    @property
    def worst_fairness_column(self) -> str:
        audits = self.fairness.get("audits", [])
        return audits[0]["column"] if audits else ""

    @property
    def n_audited(self) -> int:
        return len(self.fairness.get("columns_audited", []))

    def amplification(self, column: str) -> float | None:
        for audit in self.fairness.get("audits", []):
            if audit["column"] == column:
                return audit["gaps"].get("selection_amplification")
        return None

    def gap(self, column: str, name: str) -> float | None:
        for audit in self.fairness.get("audits", []):
            if audit["column"] == column:
                return audit["gaps"].get(name)
        return None


def _load(slug: str, display_name: str, task: str, root: Path,
          headline_keys: tuple[str, ...], fallback: dict[str, Any]
          ) -> ProjectFacts:
    where = f"{slug}/artifacts"
    selection = _read_json(root / "artifacts" / "final_model_selection.json")
    registry = _read_json(root / "models" / "model_registry.json")
    trust = root / "artifacts" / "trust"

    champion_name = _get(selection, "champion_name",
                         default=fallback["champion_name"], where=where)
    models = (registry or {}).get("models", [])

    # The champion's own parameters come from the registry, which records them;
    # final_model_selection.json does not.
    params: dict[str, Any] = {}
    for model in models:
        if model.get("is_champion"):
            params = model.get("params", {})
            break
    if not params:
        params = fallback["params"]
        STALE.append(f"{slug}/models/model_registry.json :: champion params")

    selection_metric = _get(selection, "selection_metric",
                            default=fallback["selection_metric"], where=where)

    # Runner-up is the best scorer among models of a DIFFERENT NAME.
    #
    # Two traps here, both of which produce a plausible-looking wrong answer:
    #   1. Excluding only the champion ROW is not enough - the registry holds a
    #      v002 of every model, so the champion's own second version wins the
    #      runner-up slot and the deliverable reads "Champion: Ridge /
    #      Runner-up: Ridge".
    #   2. The direction depends on the metric. Taking min() on roc_auc reports
    #      the WORST model as the runner-up, and nothing about the output looks
    #      wrong.
    higher_is_better = selection_metric not in ("rmse", "mae", "mape")
    others = [
        m for m in models
        if m.get("name") != champion_name
        and isinstance(m.get("validation_metrics", {}).get(selection_metric),
                       (int, float))
    ]
    runner: dict[str, Any] | None = None
    if others:
        chooser = max if higher_is_better else min
        runner = chooser(others,
                         key=lambda m: m["validation_metrics"][selection_metric])
    else:
        STALE.append(f"{slug}/models/model_registry.json :: runner-up")

    return ProjectFacts(
        slug=slug,
        display_name=display_name,
        task=task,
        root=root,
        champion_name=champion_name,
        champion_version=_get(selection, "champion_version",
                              default=fallback["champion_version"], where=where),
        champion_params=params,
        selection_metric=selection_metric,
        validation_metrics=_get(selection, "validation_metrics",
                                default=fallback["validation_metrics"],
                                where=where),
        test_metrics=_get(selection, "test_metrics",
                          default=fallback["test_metrics"], where=where),
        n_models=len(models) or fallback["n_models"],
        runner_up_name=(runner or {}).get("name", fallback["runner_up_name"]),
        runner_up_metrics=(runner or {}).get(
            "validation_metrics", fallback["runner_up_metrics"]),
        headline_keys=headline_keys,
        fairness=_read_json(trust / "subgroup_fairness.json") or {},
        calibration=_read_json(trust / "calibration.json") or {},
        importance=_read_json(trust / "permutation_importance.json") or {},
    )


HOUSING = _load(
    "housing", "California Housing", "Regression", PROJECT_ROOT,
    headline_keys=("rmse", "r2"),
    fallback={
        "champion_name": "Ridge",
        "champion_version": "v001",
        "params": {"alpha": 100.0},
        "selection_metric": "rmse",
        "validation_metrics": {"rmse": 64787.48, "mae": 46246.34,
                               "r2": 0.680134, "mape": 28.2869},
        "test_metrics": {"rmse": 66680.81, "mae": 48731.42,
                         "r2": 0.666513, "mape": 28.3237},
        "n_models": 6,
        "runner_up_name": "Lasso",
        "runner_up_metrics": {"rmse": 64887.4},
    },
)

CHURN = _load(
    "churn", "Telco Customer Churn", "Classification",
    PROJECT_ROOT / "workspaces" / "churn",
    headline_keys=("roc_auc", "accuracy"),
    fallback={
        "champion_name": "LogisticRegression",
        "champion_version": "v001",
        "params": {"C": 2.154434690031882},
        "selection_metric": "roc_auc",
        "validation_metrics": {"accuracy": 0.807765, "precision": 0.649805,
                               "recall": 0.596429, "f1": 0.621974,
                               "roc_auc": 0.845464},
        "test_metrics": {"accuracy": 0.810785, "precision": 0.686636,
                         "recall": 0.530249, "f1": 0.598394,
                         "roc_auc": 0.844778},
        "n_models": 6,
        "runner_up_name": "RandomForestClassifier",
        "runner_up_metrics": {"roc_auc": 0.844647},
    },
)

PROJECTS: tuple[ProjectFacts, ...] = (HOUSING, CHURN)


def warn_if_stale() -> str:
    """A message for a generator to print, or the all-clear when all read live."""
    if not STALE:
        return "All figures read live from the pipeline artifacts."
    lines = [f"WARNING: {len(STALE)} value(s) fell back to the {VERIFIED_ON} "
             f"figures because the artifact could not be read:"]
    lines += [f"  - {item}" for item in STALE]
    lines.append("Re-run the pipeline, or verify these by hand.")
    return "\n".join(lines)


def demo() -> None:
    """Self-check: the invariants every deliverable depends on."""
    for facts in PROJECTS:
        assert facts.champion_name, f"{facts.slug}: no champion"
        assert facts.runner_up_name != facts.champion_name, (
            f"{facts.slug}: runner-up ({facts.runner_up_name}) must not be "
            f"the champion")
        assert facts.test_line != "not available", f"{facts.slug}: no test line"
        assert facts.selection_metric in facts.test_metrics, (
            f"{facts.slug}: selection metric {facts.selection_metric} absent "
            f"from the test metrics")

    # Direction of the selection metric must be respected when picking a
    # runner-up: a higher-is-better metric taking min() silently reports the
    # WORST model as the runner-up, which reads as perfectly plausible.
    assert HOUSING.runner_up_metrics["rmse"] >= HOUSING.validation_metrics["rmse"], (
        "housing runner-up should not beat the champion on rmse")
    assert CHURN.runner_up_metrics["roc_auc"] <= CHURN.validation_metrics["roc_auc"], (
        "churn runner-up should not beat the champion on roc_auc")

    # The formatting contract all three deliverables share.
    assert money(66680.81) == "$66,681"
    assert metric("r2", 0.666513) == "R2 0.667"
    assert metric("roc_auc", 0.844778) == "ROC-AUC 0.8448"
    assert metric("rmse", 66680.81) == "RMSE $66,681"
    assert metric_line(None) == "not available"
    assert metric_line({}) == "not available"

    print("facts.py self-check passed.")
    for facts in PROJECTS:
        print(f"\n{facts.display_name} ({facts.task})")
        print(f"  champion   : {facts.champion_label}  {facts.tuned_parameter}")
        print(f"  validation : {facts.validation_line}")
        print(f"  test       : {facts.test_line}")
        print(f"  runner-up  : {facts.runner_up_line}")
    print(f"\n{warn_if_stale()}")


if __name__ == "__main__":
    demo()
