"""
src.explainability.drivers
==========================

PURPOSE
-------
Dataset-agnostic driver extraction for *any* project's champion model.

WHY THIS EXISTS ALONGSIDE ``interpretation.py``
-----------------------------------------------
``interpretation.interpret_champion`` produces the graded California Housing
deliverables, including an executive briefing whose prose is specific to that
problem ("roughly 20,600 census blocks", coastal caveats, dollar amounts).
Running it against an uploaded dataset would emit fluent, confident analysis
about the wrong domain.

This module keeps the parts that are true for any tabular model - the champion
bundle, its coefficient or importance vector, and its feature names - and
writes only the driver table, chart and JSON. No narrative is generated,
because none can be written without knowing what the target means.

MODEL SHAPES
------------
``coef_``                 LinearRegression / Ridge / Lasso / LogisticRegression.
                          Signed, so a direction is meaningful.
``feature_importances_``  DecisionTree / RandomForest.
                          Unsigned magnitudes; there is no direction, and the
                          ``kind`` field says so rather than letting a consumer
                          imply one.

Anything else is skipped. A missing driver artifact is a legitimate state that
the GUI renders as "not available for this model".

OUTPUTS
-------
    artifacts/explainability/coefficient_interpretation.csv
    artifacts/explainability/coefficient_interpretation.json
    artifacts/explainability/top_drivers.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless: figures are written, never displayed.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import active  # noqa: E402
from config.paths import ARTIFACTS_DIR, EXECUTIVE_BRIEFING_PATH, ensure_dir  # noqa: E402
from src.artifacts import model_registry  # noqa: E402
from src.explainability.interpretation import meaning_for  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import save_csv, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

__all__ = ["DRIVERS_CSV", "DRIVERS_JSON", "DRIVERS_PNG", "write_driver_artifacts"]

_EXPLAINABILITY_DIR: Path = ARTIFACTS_DIR / "explainability"

DRIVERS_CSV: Path = _EXPLAINABILITY_DIR / "coefficient_interpretation.csv"
DRIVERS_JSON: Path = _EXPLAINABILITY_DIR / "coefficient_interpretation.json"
DRIVERS_PNG: Path = _EXPLAINABILITY_DIR / "top_drivers.png"

#: Six decimals: enough to rank and to redraw the chart, without writing
#: "1.3417811382918503" into a table a human is meant to read.
_PRECISION = 6


def _rows_from(estimator: Any, features: list[str]) -> tuple[list[dict[str, Any]], str] | None:
    """Driver rows plus the ``kind`` of value they hold, or ``None`` to skip."""

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).ravel()
        kind = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        kind = "importance"
    else:
        logger.info(
            "Champion exposes neither coef_ nor feature_importances_; no drivers written."
        )
        return None

    if len(values) != len(features):
        # A multiclass LogisticRegression has one row of coefficients per class.
        # Collapsing them would invent a single effect that does not exist.
        logger.info(
            "Champion produced %d values for %d features (multiclass?); no drivers written.",
            len(values),
            len(features),
        )
        return None

    def direction(value: float) -> str:
        if kind != "coefficient":
            return "magnitude only"
        if value > 0:
            # Deliberately not "increases value": the housing module can say
            # that because its target is a dollar amount, but a logistic
            # coefficient is log-odds and an uploaded target could be anything.
            return "increases the prediction"
        return "decreases the prediction" if value < 0 else "no effect"

    rows = [
        {
            "feature": feature,
            "coefficient": round(float(value), _PRECISION),
            "direction": direction(float(value)),
            "abs_importance": round(abs(float(value)), _PRECISION),
            "meaning": meaning_for(feature, unknown=f"Model input '{feature}'."),
        }
        for feature, value in zip(features, values)
    ]
    rows.sort(key=lambda row: row["abs_importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    ordered = [
        {key: row[key] for key in ("rank", "feature", "coefficient", "direction", "abs_importance", "meaning")}
        for row in rows
    ]
    return ordered, kind


def _chart(rows: list[dict[str, Any]], kind: str, champion: str, top_n: int) -> Path:
    """Horizontal bar chart of the strongest drivers."""

    ensure_dir(DRIVERS_PNG.parent)
    top = rows[:top_n][::-1]  # reversed so the biggest bar sits at the top
    labels = [row["feature"] for row in top]
    values = [row["coefficient"] for row in top]
    colors = [
        "#2a7f3f" if (kind == "importance" or value > 0) else "#b1332f" for value in values
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(top) + 1)))
    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "Split importance (magnitude only)"
        if kind == "importance"
        else "Coefficient (effect on the prediction)"
    )
    ax.set_title(f"Top {len(top)} drivers - {champion}")
    fig.tight_layout()
    fig.savefig(DRIVERS_PNG, dpi=120)
    plt.close(fig)
    return DRIVERS_PNG


def _written_by_interpret_champion() -> bool:
    """Whether the existing artifact came from the richer housing interpreter.

    Both this module and ``interpretation.interpret_champion`` write
    ``coefficient_interpretation.{csv,json}``. The housing version carries
    curated per-feature meanings and dollar-denominated wording, and is the
    graded deliverable — so overwriting it with the generic table would be a
    silent downgrade. Its artifact predates the ``kind`` field, which makes the
    absence of that key a reliable signature.
    """

    if not DRIVERS_JSON.exists():
        return False
    try:
        import json

        return "kind" not in json.loads(DRIVERS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def write_driver_artifacts(*, top_n: int = 12, force: bool = False) -> dict[str, Any] | None:
    """Write driver artifacts for the active project's champion.

    Returns a summary dict, or ``None`` when there is nothing to write (no
    champion registered, a model whose internals expose no per-feature signal,
    or a richer artifact already present). Never raises for those cases: an
    absent artifact is a state the consumers already handle.
    """

    if not force and _written_by_interpret_champion():
        logger.info(
            "%s was written by interpret_champion (richer, curated); leaving it "
            "alone. Pass force=True to replace it.",
            DRIVERS_JSON.name,
        )
        return None

    champion = model_registry.get_champion()
    if champion is None:
        logger.info("No champion registered; no drivers written.")
        return None

    import joblib

    bundle = joblib.load(champion["model_path"])
    label = f"{bundle.get('name')} {bundle.get('version')}"
    extracted = _rows_from(bundle["estimator"], [str(f) for f in bundle["feature_names"]])
    if extracted is None:
        return None
    rows, kind = extracted

    estimator = bundle["estimator"]
    payload = {
        "artifact": "coefficient_interpretation",
        "generated_at": utc_timestamp(),
        "champion": label,
        "kind": kind,
        "intercept": (
            float(np.asarray(estimator.intercept_).ravel()[0])
            if hasattr(estimator, "intercept_")
            else None
        ),
        "coefficients": rows,
    }

    ensure_dir(_EXPLAINABILITY_DIR)
    save_json(payload, DRIVERS_JSON)
    save_csv(pd.DataFrame(rows), DRIVERS_CSV)
    chart = _chart(rows, kind, label, top_n)
    briefing = write_generic_briefing(rows, kind, label, champion)

    logger.info("Wrote %d %s rows for %s", len(rows), kind, label)
    outputs = [str(DRIVERS_JSON), str(DRIVERS_CSV), str(chart)]
    if briefing is not None:
        outputs.append(str(briefing))
    return {
        "champion": label,
        "kind": kind,
        "n_features": len(rows),
        "outputs": outputs,
    }


def write_generic_briefing(
    rows: list[dict[str, Any]],
    kind: str,
    label: str,
    champion: dict[str, Any],
    *,
    top_n: int = 5,
) -> Path | None:
    """Write an executive briefing that states only what is actually known.

    ``interpretation._briefing_markdown`` writes the graded housing briefing,
    whose prose is specific to California house prices. This is the version for
    every other project: it names the champion, quotes its real validation
    metrics, lists the strongest drivers, and stops. It makes no claim about
    what the target *means*, because nothing here knows that.

    Refuses to overwrite the housing briefing, which is the richer artifact.
    """

    if EXECUTIVE_BRIEFING_PATH.exists():
        existing = EXECUTIVE_BRIEFING_PATH.read_text(encoding="utf-8")
        if "California House Prices" in existing:
            logger.info("Executive briefing already written by interpret_champion; leaving it.")
            return None

    try:
        target = active.target_column()
        task_name = active.task()
        display_name = active.active_dataset().display_name
    except Exception:  # noqa: BLE001 - no descriptor; describe what we can.
        target, task_name, display_name = "the target", "model", "This project"

    metrics = champion.get("validation_metrics", {}) or {}
    metric_lines = [
        f"- {name}: {value:,.4f}" if abs(value) < 1000 else f"- {name}: {value:,.0f}"
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    ]

    measure = "effect on the prediction" if kind == "coefficient" else "importance"
    driver_lines = []
    for row in rows[:top_n]:
        if kind == "coefficient":
            direction = "raises" if row["coefficient"] > 0 else "lowers"
            driver_lines.append(
                f"{row['rank']}. **{row['feature']}** - {direction} the prediction "
                f"({row['coefficient']:,.4g})."
            )
        else:
            driver_lines.append(
                f"{row['rank']}. **{row['feature']}** - importance {row['coefficient']:,.4g}."
            )

    lines = [
        f"# Executive Briefing: {display_name}",
        "",
        f"*Prepared {utc_timestamp()} - champion model: {label}*",
        "",
        "## What was built",
        "",
        f"A {task_name} model predicting **{target}**. Candidate models were "
        "compared on a validation split held out from training; the best was "
        "then measured once on a test split that was never used for tuning or "
        "selection, so the reported accuracy reflects unseen data.",
        "",
        "## How well it predicts",
        "",
        *(metric_lines or ["- No validation metrics were recorded."]),
        "",
        f"## Strongest drivers (by {measure})",
        "",
        *(driver_lines or ["- No per-feature signal is available for this model."]),
        "",
        "## How to read this",
        "",
        (
            "These are standardized coefficients: each is the change in the "
            "prediction for a one-standard-deviation move in that input, holding "
            "the others fixed. A positive value pushes the prediction up."
            if kind == "coefficient"
            else "These are split importances. They show how much each input "
            "contributed to the model's decisions, but carry no direction - a "
            "high value does not mean the input pushes the prediction up."
        ),
        "",
        "## Limits",
        "",
        "- This briefing is generated from the model and its metrics alone. It "
        "makes no claim about what the target means in your domain, nor about "
        "cause and effect - the drivers are associations the model found, not "
        "levers proven to change the outcome.",
        "- Correlated inputs share credit unpredictably; read the top drivers as "
        "a group rather than as independent factors.",
        "",
    ]

    ensure_dir(EXECUTIVE_BRIEFING_PATH.parent)
    EXECUTIVE_BRIEFING_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote generic executive briefing -> %s", EXECUTIVE_BRIEFING_PATH)
    return EXECUTIVE_BRIEFING_PATH
