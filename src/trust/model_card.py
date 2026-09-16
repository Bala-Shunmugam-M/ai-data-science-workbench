"""
src.trust.model_card
====================

PURPOSE
-------
Assemble one document that answers "what is this model, and can I rely on it?"
from the artifacts the pipeline has already written.

WHY THIS IS MOSTLY ASSEMBLY
---------------------------
Almost everything a model card needs is already on disk by the time this runs -
the registry knows the version and hyperparameters, the evaluation report knows
the metrics and why this model won, the approvals know who signed off, the audit
log and lineage know what happened in what order. What was missing was a single
place that reads them together and says so in one page. This module computes
almost nothing; it reads, arranges, and refuses to paper over gaps.

THE ONE RULE
------------
A section with no evidence says so, in the card, with the reason. It is never
silently dropped. A card that omits its fairness section looks like a card whose
fairness was fine, and that is precisely the failure this artifact exists to
prevent. The ``sources`` block at the end lists every artifact this card looked
for and whether it was found, so a reader can tell an incomplete pipeline from a
clean bill of health.

WHAT THIS MODULE WILL NOT WRITE
-------------------------------
An intended-use statement. Every model-card standard asks for one, and it cannot
be derived from any artifact here - it is a claim about which decisions the model
is allowed to inform, which only a person who knows the domain can make. The card
therefore states that it is absent and who must write it, rather than generating
fluent prose about a use case nobody approved.

OUTPUTS
-------
    artifacts/trust/model_card.md
    artifacts/trust/model_card.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import active
from config.paths import (
    AUDIT_LOG_PATH,
    CALIBRATION_JSON_PATH,
    ENGINEERED_TEST_PATH,
    ENGINEERED_TRAIN_PATH,
    ENGINEERED_VALIDATION_PATH,
    EXPERIMENTS_LOG_PATH,
    EXPLAINABILITY_DIR,
    FAIRNESS_JSON_PATH,
    FEATURE_APPROVAL_PATH,
    LINEAGE_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_CARD_JSON_PATH,
    MODEL_CARD_MD_PATH,
    MODEL_EVALUATION_REPORT_PATH,
    ROBUST_IMPORTANCE_JSON_PATH,
    ensure_dir,
)
from src.artifacts import model_registry
from src.utils.common import utc_timestamp
from src.utils.file_utils import save_json, save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["build", "run"]

_DRIVERS_JSON: Path = EXPLAINABILITY_DIR / "coefficient_interpretation.json"

#: Every artifact the card reads, and the section that goes quiet without it.
_SOURCES: dict[str, Path] = {
    "evaluation_report": MODEL_EVALUATION_REPORT_PATH,
    "drivers": _DRIVERS_JSON,
    "permutation_importance": ROBUST_IMPORTANCE_JSON_PATH,
    "subgroup_fairness": FAIRNESS_JSON_PATH,
    "calibration": CALIBRATION_JSON_PATH,
    "feature_approval": FEATURE_APPROVAL_PATH,
    "model_approval": MODEL_APPROVAL_PATH,
    "audit_log": AUDIT_LOG_PATH,
    "lineage": LINEAGE_PATH,
    "experiments": EXPERIMENTS_LOG_PATH,
}


def _read_json(path: Path) -> Any | None:
    """Parse ``path``, or ``None`` when it is absent or unreadable.

    An unreadable artifact is treated the same as a missing one *for the card's
    content*, but the distinction survives in the ``sources`` block: ``exists``
    is true while the section still reports no evidence, which is the signature of
    a corrupt artifact rather than an unrun stage.
    """

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not parse %s; treating it as absent.", path.name)
        return None


def _count_lines(path: Path) -> int | None:
    """Row count of a JSONL file, or ``None`` when it is absent."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _count_rows(path: Path) -> int | None:
    """Data-row count of a CSV (header excluded), or ``None`` when absent."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        total = sum(1 for _ in handle)
    return max(0, total - 1)


def _num(value: Any, spec: str | None = None, missing: str = "not recorded") -> str:
    """Format a number, or say it is missing.

    Every section here reads values out of artifacts that may predate the field
    being read - the housing driver table was written before ``kind`` existed, and
    a degenerate split can leave a calibration error undefined. A bare
    ``f"{value:,.4f}"`` raises ``TypeError`` on ``None`` and takes the whole card
    down with it, so no section is allowed to format a number directly.

    With no ``spec``, magnitude picks the notation. ``,.4g`` alone renders a
    regression gap of 29,104 dollars as ``2.91e+04``, which no reader converts in
    their head - so anything at or above a thousand switches to grouped fixed
    notation, matching :func:`_format_metric` below.
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return missing
    if spec is None:
        spec = ",.0f" if abs(value) >= 1000 else ",.4g"
    return format(value, spec)


def _format_metric(name: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        return f"{name}: {value}"
    return f"{name}: {value:,.4f}" if abs(value) < 1000 else f"{name}: {value:,.2f}"


def _metric_rows(metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return ["- No metrics recorded."]
    return [f"- {_format_metric(name, value)}" for name, value in metrics.items()]


def _dataset_facts() -> dict[str, Any]:
    """Target, task and display name from the active project's descriptor."""

    try:
        descriptor = active.active_dataset()
        return {
            "display_name": descriptor.display_name,
            "target": descriptor.target,
            "task": descriptor.task,
            "selection_metric": descriptor.selection_metric,
        }
    except Exception as exc:  # noqa: BLE001 - a card is still worth writing.
        logger.warning("No dataset descriptor for the active project: %s", exc)
        return {
            "display_name": "Unnamed project",
            "target": None,
            "task": None,
            "selection_metric": None,
        }


def build() -> dict[str, Any] | None:
    """
    Gather every input for the card into one dict, or ``None`` with no champion.

    Separated from :func:`run` so the assembled structure can be inspected and
    tested without writing files.
    """

    champion = model_registry.get_champion()
    if champion is None:
        logger.info("No champion registered; no model card written.")
        return None

    evaluation = _read_json(MODEL_EVALUATION_REPORT_PATH) or {}
    facts = _dataset_facts()

    # The identity comes from the registry and the metrics come from the evaluation
    # report, and nothing forces them to describe the same model. Train and register
    # anything between `evaluate` and `trust` and the card would print model X's
    # name, version and data_hash above model Y's test metrics - the most
    # consequential possible failure in a document whose rule is "a section with no
    # evidence says so". Both artifacts carry name and version, so this is checkable.
    evaluated = evaluation.get("champion") or {}
    metrics_stale = bool(evaluated) and (
        evaluated.get("name") != champion.get("name")
        or evaluated.get("version") != champion.get("version")
    )
    if metrics_stale:
        logger.warning(
            "Registry champion is %s %s but the evaluation report describes %s %s; "
            "the card will report the metrics as stale rather than pair them.",
            champion.get("name"),
            champion.get("version"),
            evaluated.get("name"),
            evaluated.get("version"),
        )

    return {
        "artifact": "model_card",
        "generated_at": utc_timestamp(),
        "project": facts,
        "model": {
            "name": champion.get("name"),
            "version": champion.get("version"),
            "registered_at": champion.get("registered_at"),
            "params": champion.get("params") or {},
            "data_hash": champion.get("data_hash"),
            "model_path": champion.get("model_path"),
        },
        "intended_use": None,  # See the module docstring: not derivable, not invented.
        "data": {
            "train_rows": _count_rows(ENGINEERED_TRAIN_PATH),
            "validation_rows": _count_rows(ENGINEERED_VALIDATION_PATH),
            "test_rows": _count_rows(ENGINEERED_TEST_PATH),
        },
        "performance": {
            "selection_metric": evaluation.get("selection_metric")
            or facts["selection_metric"],
            "n_models_compared": evaluation.get("n_models_compared"),
            "validation_metrics": (
                champion.get("validation_metrics")
                if metrics_stale
                else evaluated.get("validation_metrics")
                or champion.get("validation_metrics")
            ),
            # Withheld rather than shown against the wrong model. The registry
            # carries no test metrics by design (the test split is scored once, by
            # the evaluator), so there is no substitute to fall back to.
            "test_metrics": None if metrics_stale else evaluated.get("test_metrics"),
            "selection_narrative": (
                None if metrics_stale else evaluation.get("selection_narrative")
            ),
            "metrics_stale": metrics_stale,
            "evaluated_model": (
                f"{evaluated.get('name')} {evaluated.get('version')}"
                if metrics_stale
                else None
            ),
        },
        "drivers": _read_json(_DRIVERS_JSON),
        "permutation_importance": _read_json(ROBUST_IMPORTANCE_JSON_PATH),
        "fairness": _read_json(FAIRNESS_JSON_PATH),
        "calibration": _read_json(CALIBRATION_JSON_PATH),
        "governance": {
            "feature_approval": _read_json(FEATURE_APPROVAL_PATH),
            "model_approval": _read_json(MODEL_APPROVAL_PATH),
            "audit_events": _count_lines(AUDIT_LOG_PATH),
            "experiment_runs": _count_lines(EXPERIMENTS_LOG_PATH),
            "lineage_nodes": len((_read_json(LINEAGE_PATH) or {}).get("nodes", []) or []),
        },
        "sources": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in _SOURCES.items()
        },
    }


def _drivers_section(card: dict[str, Any]) -> list[str]:
    """Model-internal drivers and permutation importance, side by side."""

    lines: list[str] = ["## What the model depends on", ""]

    drivers = card.get("drivers")
    if drivers and drivers.get("coefficients"):
        kind = drivers.get("kind", "coefficient")
        measure = "effect on the prediction" if kind == "coefficient" else "split importance"
        lines += [f"**From the model's own internals** ({measure}):", ""]
        for row in drivers["coefficients"][:8]:
            lines.append(
                f"{row.get('rank')}. `{row.get('feature')}` — "
                f"{_num(row.get('coefficient'))} ({row.get('direction', 'n/a')})"
            )
        lines.append("")
    else:
        lines += [
            "**From the model's own internals:** not available. The champion "
            "exposes neither coefficients nor feature importances, or the "
            "explainability stage has not run.",
            "",
        ]

    importance = card.get("permutation_importance")
    if importance and importance.get("importances"):
        lines += [
            f"**By permutation** (drop in {importance.get('measured_against')} when a "
            f"column is shuffled, {importance.get('n_repeats')} shuffles, measured on "
            f"the {importance.get('split')} split):",
            "",
        ]
        for row in importance["importances"][:8]:
            marker = "" if row["distinguishable_from_noise"] else "  *(not separable from noise)*"
            lines.append(
                f"{row['rank']}. `{row['feature']}` — "
                f"{_num(row.get('importance_mean'))} ± "
                f"{_num(row.get('importance_std'), ',.3g')}{marker}"
            )
        separable = importance.get("n_distinguishable_from_noise")
        total = importance.get("n_features")
        lines += [
            "",
            f"{separable} of {total} features show a dependence larger than the "
            "measurement's own noise.",
            "",
            "Where the two lists disagree, the disagreement is the finding: a "
            "feature with a large coefficient and no permutation importance is one "
            "the model cannot actually use on unseen data.",
            "",
        ]
    else:
        lines += [
            "**By permutation:** not assessed — the trust stage has not run, or it "
            "produced no importances.",
            "",
        ]
    return lines


#: The conventional adverse-impact threshold - a US EEOC enforcement convention
#: (the "four-fifths rule"), not a statutory limit and not a definition of
#: fairness. It is cited because a ratio needs some reference point to be legible,
#: and this is the one a reader is most likely to have met.
_FOUR_FIFTHS = 0.8


def _ratio_note(name: str, value: Any) -> str:
    """Annotate a selection-rate ratio so its severity is not lost in a list.

    A ratio of exactly 0 means one whole subgroup is never predicted positive -
    the strongest disparate-impact result this artifact can produce - and it
    renders as a bare "0" alongside nine other numbers unless it is called out.
    """

    if name != "selection_rate_ratio" or not isinstance(value, (int, float)):
        return ""
    if value == 0:
        return (
            " — **one group is never predicted positive at all.** Whatever the "
            "positive prediction triggers, that group cannot receive it"
        )
    if value < _FOUR_FIFTHS:
        return (
            f" — below the conventional four-fifths ({_FOUR_FIFTHS:.0%}) reference "
            "point for adverse impact"
        )
    return ""


def _fairness_section(card: dict[str, Any]) -> list[str]:
    """Subgroup gaps, or a stated reason there are none to show."""

    lines: list[str] = ["## Subgroup performance", ""]
    fairness = card.get("fairness")

    if not fairness:
        lines += [
            "**Not assessed.** No subgroup fairness artifact was found; the trust "
            "stage has not run for this project.",
            "",
        ]
        return lines

    audits = fairness.get("audits") or []
    if not audits:
        lines += [
            "**No subgroup column found.** No column in the "
            f"{fairness.get('split')} split had between 2 and "
            f"{fairness.get('max_levels')} distinct non-float values, so there is "
            "nothing to compare groups across. This is a property of the dataset, "
            "not a clean result — the model has not been shown to perform evenly "
            "across any segment.",
            "",
        ]
        return lines

    lines += [
        f"Measured on the {fairness.get('split')} split "
        f"({_num(fairness.get('n_rows'), ',.0f', missing='row count not recorded')} "
        f"rows). Groups smaller than "
        f"{_num(fairness.get('min_group_size'), ',.0f')} rows are reported but "
        "excluded from the gap arithmetic.",
        "",
    ]
    for audit in audits:
        gaps = audit.get("gaps", {})
        lines.append(
            f"### `{audit['column']}` — {audit['n_groups']} groups, "
            f"{gaps.get('n_groups_compared')} compared"
        )
        lines.append("")
        if audit.get("gaps_measurable") is False:
            # Small split: the per-group rates below are real, but no pair of groups
            # was large enough to compare, so there is no disparity finding either way.
            lines += [
                "**No gap was measurable for this column** — every group fell below "
                f"the {fairness.get('min_group_size')}-row minimum. The per-group "
                "rates are still reported, but this column shows neither a disparity "
                "nor the absence of one.",
                "",
            ]
        reported = [
            f"- {name.replace('_', ' ')}: **{_num(value)}**{_ratio_note(name, value)}"
            for name, value in gaps.items()
            if name.endswith(("_gap", "_ratio")) and isinstance(value, (int, float))
        ]
        undefined = [
            name.replace("_", " ")
            for name, value in gaps.items()
            if name.endswith(("_gap", "_ratio")) and value is None
        ]
        lines += reported or ["- No gap was measurable across these groups."]
        if undefined:
            lines.append(
                f"- Not measurable here: {', '.join(undefined)} — fewer than two "
                "groups had both a defined value and enough rows."
            )
        amplification = gaps.get("selection_amplification")
        if isinstance(amplification, (int, float)):
            # The number that separates "the model is biased" from "the model is
            # right": how much wider the model spreads these groups than their
            # actual outcomes are spread.
            reading = (
                "the model spreads these groups further apart than their actual "
                "outcomes are"
                if amplification > 0.02
                else "the model under-separates groups that genuinely differ"
                if amplification < -0.02
                else "the model tracks the real difference between these groups"
            )
            lines.append(
                f"- selection amplification: **{_num(amplification, '+,.4g')}** — {reading}"
            )
        for excluded in audit.get("excluded_from_gaps") or []:
            lines.append(
                f"- Excluded from the gaps: `{excluded['group']}` "
                f"(n={excluded['n']}, {excluded['reason']})"
            )
        lines.append("")

    lines += [
        "A gap is not a verdict. Whether one is acceptable depends on the domain, "
        "the decision the prediction feeds, and the law that applies — none of "
        "which this pipeline knows.",
        "",
    ]
    # Only where amplification exists to explain. This paragraph names the kind of
    # column it applies to, and pasting it onto a regression card produced
    # confident prose about "contract types" for a house-price model - the same
    # wrong-domain failure `drivers.py` refuses the housing briefing to avoid.
    if any(audit["metric_set"] == "binary_classification" for audit in audits):
        lines += [
            "Every low-cardinality column is audited, including columns the model "
            "uses as features on purpose. A large gap across such a column is "
            "usually the model being accurate; `selection amplification` is the "
            "part it adds beyond the difference that is genuinely there.",
            "",
        ]
    return lines


def _calibration_section(card: dict[str, Any]) -> list[str]:
    """Probability reliability, or the stated reason it does not apply."""

    lines: list[str] = ["## Probability calibration", ""]
    calibration = card.get("calibration")

    if not calibration:
        lines += [
            "**Not assessed.** No calibration artifact was found; the trust stage "
            "has not run for this project.",
            "",
        ]
        return lines
    if not calibration.get("assessed"):
        lines += [f"**Not applicable.** {calibration.get('reason')}", ""]
        return lines

    ece = calibration.get("ece")
    lines += [
        f"- Brier score: **{_num(calibration.get('brier_score'), ',.4f')}**",
        f"- Expected calibration error (ECE): **{_num(ece, ',.4f')}**"
        + (
            f" — about {ece * 100:.0f} points off on average"
            if isinstance(ece, (int, float))
            else ""
        ),
        f"- Worst band (MCE): **{_num(calibration.get('mce'), ',.4f')}**",
        f"- Mean predicted probability {_num(calibration.get('mean_predicted'), ',.4f')} vs "
        f"actual base rate {_num(calibration.get('base_rate'), ',.4f')} "
        f"(bias {_num(calibration.get('global_bias'), '+,.4f')})",
        "",
        "Calibration is separate from ranking quality. ROC-AUC measures whether "
        "the model orders cases correctly; this measures whether its probabilities "
        "mean what they say. Anything that multiplies a predicted probability by a "
        "cost depends on this number, not on the AUC.",
        "",
    ]
    return lines


def render(card: dict[str, Any]) -> str:
    """Render the assembled card as markdown."""

    project = card["project"]
    model = card["model"]
    performance = card["performance"]
    data = card["data"]
    governance = card["governance"]

    lines: list[str] = [
        f"# Model Card: {project['display_name']}",
        "",
        f"*Generated {card['generated_at']} — "
        f"{model['name']} {model['version']}*",
        "",
        "## Model details",
        "",
        f"- Family: **{model['name']}**",
        f"- Version: **{model['version']}**, registered {model['registered_at']}",
        f"- Task: **{project['task'] or 'unknown'}**, "
        f"target **{project['target'] or 'unknown'}**",
        f"- Hyperparameters: "
        + (
            ", ".join(f"`{k}={v}`" for k, v in model["params"].items())
            if model["params"]
            else "defaults (none tuned)"
        ),
        f"- Training-data fingerprint: `{model['data_hash'] or 'not recorded'}`",
        "",
        "## Intended use",
        "",
        "**Not authored.** This card records what the model is and how it behaves. "
        "It does not state which decisions the model may be used for, because that "
        "is not derivable from any artifact in this pipeline — it is a judgement "
        "about the domain and the consequences of a wrong prediction. Whoever "
        "deploys this model owns that statement.",
        "",
        "## Data",
        "",
        f"- Train: {data['train_rows']:,} rows"
        if data["train_rows"] is not None
        else "- Train: not found",
        f"- Validation: {data['validation_rows']:,} rows"
        if data["validation_rows"] is not None
        else "- Validation: not found",
        f"- Test: {data['test_rows']:,} rows (used once, after the champion was chosen)"
        if data["test_rows"] is not None
        else "- Test: not found",
        "",
        "## Performance",
        "",
        f"Champion selected by **{performance['selection_metric'] or 'an unrecorded metric'}** "
        "on the validation split"
        + (
            f", from {performance['n_models_compared']} candidate"
            f"{'' if performance['n_models_compared'] == 1 else 's'}."
            if isinstance(performance["n_models_compared"], int)
            else ". The number of candidates compared was not recorded."
        ),
        "",
        "**Validation:**",
        *_metric_rows(performance["validation_metrics"]),
        "",
        "**Test (held out until the champion was fixed):**",
        *_metric_rows(performance["test_metrics"]),
        "",
    ]
    if performance.get("metrics_stale"):
        lines[-1:] = [
            "",
            f"> **The evaluation report describes a different model** "
            f"(`{performance.get('evaluated_model')}`) than the registry champion "
            f"above. Its test metrics are withheld rather than shown against this "
            f"model. Re-run `python main.py evaluate` before trusting this section.",
            "",
        ]
    if performance.get("selection_narrative"):
        lines += [f"> {performance['selection_narrative']}", ""]

    lines += _drivers_section(card)
    lines += _fairness_section(card)
    lines += _calibration_section(card)

    approvals_present = [
        name
        for name in ("feature_approval", "model_approval")
        if card["governance"].get(name)
    ]
    lines += [
        "## Governance trail",
        "",
        f"- Approvals on file: "
        + (", ".join(f"`{name}`" for name in approvals_present) if approvals_present else "**none**"),
        f"- Audit events recorded: {governance['audit_events'] if governance['audit_events'] is not None else 'no audit log'}",
        f"- Training runs logged: {governance['experiment_runs'] if governance['experiment_runs'] is not None else 'no experiment log'}",
        f"- Lineage nodes: {governance['lineage_nodes']}",
        "",
        "## Limitations",
        "",
        "- Every number here is measured on one dataset and one split. They are "
        "estimates with sampling error, not guarantees.",
        "- The drivers and importances are associations the model found. None of "
        "them is evidence that changing the input would change the outcome.",
        "- Nothing in this pipeline monitors the model after this point. These "
        "numbers describe the data as it was when the model was trained; they say "
        "nothing about how it will behave once that distribution moves.",
        "",
        "## Sources",
        "",
        "Every artifact this card looked for. A section reporting \"not assessed\" "
        "next to a `found` source means the artifact exists but is empty or "
        "unreadable.",
        "",
    ]
    for name, source in card["sources"].items():
        lines.append(f"- `{name}`: {'found' if source['exists'] else '**missing**'}")
    lines.append("")

    return "\n".join(lines)


def run() -> dict[str, Any] | None:
    """Write ``model_card.md`` and ``model_card.json``, or ``None`` with no champion."""

    card = build()
    if card is None:
        return None

    ensure_dir(MODEL_CARD_MD_PATH.parent)
    save_text(render(card), MODEL_CARD_MD_PATH)
    save_json(card, MODEL_CARD_JSON_PATH)

    missing = [name for name, source in card["sources"].items() if not source["exists"]]
    logger.info(
        "Model card written for %s %s (%d of %d sources found).",
        card["model"]["name"],
        card["model"]["version"],
        len(card["sources"]) - len(missing),
        len(card["sources"]),
    )
    if missing:
        logger.info("Sections reporting no evidence: %s", ", ".join(missing))
    return card
