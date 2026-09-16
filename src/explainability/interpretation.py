"""
src.explainability.interpretation
=================================

PURPOSE
-------
Interpret the champion linear-family model (LinearRegression / Ridge / Lasso) for
a managerial audience. Because the modelling matrix standardizes every numeric
predictor (train mean/std) and one-hot encodes ``ocean_proximity``, each fitted
numeric coefficient is already a *standardized* effect: the dollar change in
predicted median house value for a one-standard-deviation increase in that
predictor, holding the others fixed. One-hot coefficients are the dollar effect
of belonging to that coastal-proximity category versus the all-zero baseline.

This module produces, for the champion:

    1. A standardized-coefficients table (feature, coefficient, direction,
       absolute importance, rank, plain-English meaning).
    2. A top-drivers horizontal bar chart PNG.
    3. A one-page, non-mathematical executive briefing in Markdown.

PIPELINE POSITION
-----------------
    evaluation -> [explainability] -> reporting

OUTPUTS
-------
    artifacts/explainability/coefficient_interpretation.csv
    artifacts/explainability/coefficient_interpretation.json
    artifacts/explainability/top_drivers.png
    artifacts/reports/executive_briefing.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are written, never displayed.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config.constants import TARGET_COLUMN, TARGET_VALUE_CAP  # noqa: E402
from config.paths import (  # noqa: E402
    EXECUTIVE_BRIEFING_PATH,
    EXPLAINABILITY_DIR,
    ensure_dir,
)
from src.artifacts import model_registry  # noqa: E402
from src.governance import GovernanceError, audit_logger, lineage_tracker  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import save_csv, save_json, save_text  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

# Linear-family estimators this interpreter supports (all expose ``coef_``).
_LINEAR_FAMILY = {"LinearRegression", "Ridge", "Lasso"}

COEFFICIENT_CSV: Path = EXPLAINABILITY_DIR / "coefficient_interpretation.csv"
COEFFICIENT_JSON: Path = EXPLAINABILITY_DIR / "coefficient_interpretation.json"
TOP_DRIVERS_PNG: Path = EXPLAINABILITY_DIR / "top_drivers.png"

# Plain-English descriptions of engineered / base design-matrix features. Any
# feature not listed falls back to a generic template so the table is always
# complete even if the approved feature set changes.
_FEATURE_MEANINGS: dict[str, str] = {
    "median_income": "District median household income (the dominant driver of value).",
    "housing_median_age": "Typical age of the housing stock in the district.",
    "total_rooms": "Total rooms across all homes in the census block.",
    "total_bedrooms": "Total bedrooms across all homes in the census block.",
    "population": "Resident population of the census block.",
    "households": "Number of households in the census block.",
    "longitude": "East-west location (proxy for coastal/inland position).",
    "latitude": "North-south location (proxy for region within California).",
    "rooms_per_household": "Average home size (rooms per household); a wealth/space proxy.",
    "bedrooms_per_room": "Bedroom share of rooms; higher values flag smaller/denser units.",
    "population_per_household": "Average household occupancy (crowding).",
    "bedrooms_per_household": "Average bedrooms available per household.",
    "rooms_per_person": "Living space per resident.",
    "bedrooms_per_person": "Bedrooms available per resident.",
    "log_total_rooms": "Scale of the housing stock (skew-compressed total rooms).",
    "log_total_bedrooms": "Scale of bedrooms (skew-compressed total bedrooms).",
    "log_population": "Scale of population (skew-compressed).",
    "log_households": "Scale of households (skew-compressed).",
}

_OCEAN_MEANINGS: dict[str, str] = {
    "<1H OCEAN": "Home is within an hour of the ocean (premium coastal access).",
    "INLAND": "Home is inland (typically the lowest-value segment).",
    "ISLAND": "Home is on an island (rare, high-value segment).",
    "NEAR BAY": "Home is near the bay (established, higher-value metro areas).",
    "NEAR OCEAN": "Home is near the ocean (coastal amenity premium).",
}


#: Default wording when nothing curated matches. Correct for this project's
#: dollar-valued target; callers analysing another dataset should pass their
#: own ``unknown`` rather than inherit it (see :func:`meaning_for`).
UNKNOWN_FEATURE_TEMPLATE = "Effect of predictor '{feature}' on predicted value."


def meaning_for(feature: str, *, unknown: str | None = None) -> str:
    """Plain-English meaning for a design-matrix feature name.

    ``unknown`` overrides the wording used when no curated meaning matches.
    It exists because this module's default says "on predicted value", which is
    true for the housing target and wrong for anything else — a logistic
    coefficient is log-odds, and an uploaded dataset's target could be
    anything. Callers outside the housing pipeline (``webapi/bridge.py``) pass
    their own neutral phrasing rather than inheriting a claim about dollars.
    """

    if feature in _FEATURE_MEANINGS:
        return _FEATURE_MEANINGS[feature]
    if feature.startswith("ocean_proximity_"):
        level = feature[len("ocean_proximity_"):]
        return _OCEAN_MEANINGS.get(
            level, f"Coastal-proximity indicator for the '{level}' category."
        )
    return unknown if unknown is not None else UNKNOWN_FEATURE_TEMPLATE.format(feature=feature)


def load_champion_bundle() -> dict[str, Any]:
    """Load the champion model bundle from the registry, or raise clearly."""

    champion = model_registry.get_champion()
    if champion is None:
        raise GovernanceError(
            "No champion model is set. Run 'python main.py evaluate' first."
        )
    bundle = joblib.load(champion["model_path"])
    bundle["registry_entry"] = champion
    return bundle


def build_coefficient_table(bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Build the standardized-coefficient interpretation table for the champion.

    Numeric coefficients are standardized (per-1-SD dollar effects); one-hot
    coefficients are per-category dollar effects versus the baseline. Rows are
    ranked by absolute coefficient magnitude (1 = most influential).
    """

    name = bundle["name"]
    if name not in _LINEAR_FAMILY:
        raise GovernanceError(
            f"Champion '{name}' is not a linear-family model; coefficient "
            "interpretation supports LinearRegression / Ridge / Lasso only."
        )

    estimator = bundle["estimator"]
    feature_names = list(bundle["feature_names"])
    coefficients = [float(c) for c in estimator.coef_]

    rows: list[dict[str, Any]] = []
    for feature, coef in zip(feature_names, coefficients):
        rows.append(
            {
                "feature": feature,
                "coefficient": coef,
                "direction": "increases value"
                if coef > 0
                else ("decreases value" if coef < 0 else "no effect"),
                "abs_importance": abs(coef),
                "meaning": meaning_for(feature),
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values("abs_importance", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def _top_drivers_chart(table: pd.DataFrame, top_n: int, path: Path) -> Path:
    """Write a horizontal bar chart of the top-N standardized coefficients."""

    ensure_dir(path.parent)
    top = table.head(top_n).iloc[::-1]  # reverse so the biggest bar is on top
    colors = ["#2a7f3f" if c > 0 else "#b1332f" for c in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(top) + 1)))
    ax.barh(top["feature"], top["coefficient"], color=colors, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (dollar change per 1 SD / category)")
    ax.set_title(
        f"Top {len(top)} value drivers - {table.attrs.get('champion_label', 'champion')}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _briefing_markdown(
    bundle: dict[str, Any],
    table: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """Compose the one-page, non-mathematical executive briefing."""

    entry = bundle["registry_entry"]
    name = bundle["name"]
    version = bundle["version"]
    val = entry.get("validation_metrics", {})
    intercept = float(bundle["estimator"].intercept_)

    top = table.head(top_n)
    driver_lines = []
    for _, r in top.iterrows():
        arrow = "raises" if r["coefficient"] > 0 else "lowers"
        driver_lines.append(
            f"{int(r['rank'])}. **{r['feature']}** - {r['meaning']} "
            f"Each one-standard-deviation move {arrow} predicted median value by "
            f"about ${abs(r['coefficient']):,.0f}."
        )

    lines = [
        "# Executive Briefing: What Drives California House Prices",
        "",
        f"*Prepared {utc_timestamp()} - champion model: {name} {version}*",
        "",
        "## Business question",
        "",
        "Which measurable characteristics of a California census block most affect "
        "its median home value, and can we predict that value reliably enough to "
        "support portfolio, pricing, and market-entry decisions?",
        "",
        "## The data",
        "",
        "The analysis uses the California census housing dataset: roughly 20,600 "
        "census blocks, each described by location, housing age, room and bedroom "
        f"counts, population, households, median income, coastal proximity, and the "
        f"target, **{TARGET_COLUMN}**. The data was split into training, validation, "
        "and an untouched test set; every statistic used to clean and scale the data "
        "was learned from the training portion only, so the reported accuracy is an "
        "honest estimate of performance on unseen blocks.",
        "",
        "## Method, in one paragraph",
        "",
        "We fit a family of transparent linear models in which every input is placed "
        "on a common scale, so each model produces one easily-read number per factor: "
        "how many dollars the predicted value moves when that factor increases by a "
        "typical amount. We compared the candidates on the validation set, selected "
        f"the best performer (**{name}**), and measured it once on the held-out test "
        "set. No opaque black-box techniques were used, so every prediction can be "
        "explained to a stakeholder in plain business terms.",
        "",
        "## How well it predicts",
        "",
        f"- Validation error (RMSE): about ${val.get('rmse', float('nan')):,.0f}",
        f"- Validation R-squared: {val.get('r2', float('nan')):.3f} "
        "(share of value variation explained)",
        f"- Baseline predicted value (intercept): about ${intercept:,.0f}",
        "",
        "## Top 5 value drivers",
        "",
        *driver_lines,
        "",
        "## Important caveats",
        "",
        "- **Multicollinearity among room/household counts.** Total rooms, bedrooms, "
        "population, and households move together; their individual coefficients "
        "should be read as a group, not as independent levers.",
        "- **Coastal nonlinearity.** The value premium for coastal proximity is not a "
        "straight line; the linear model approximates it with category indicators and "
        "will understate sharp coastal effects.",
        "- **Census-block aggregation.** Every figure is a block-level average, not an "
        "individual house; conclusions apply to neighbourhoods, not single properties.",
        f"- **Target cap.** Median values are capped at ~${TARGET_VALUE_CAP:,.0f} in the "
        "source census data, so the model cannot distinguish the most expensive blocks "
        "from one another and will under-predict at the very top of the market.",
        "",
        "## Recommendations",
        "",
        "1. Treat **median income** and **coastal proximity** as the primary levers "
        "when screening markets; they dominate predicted value.",
        "2. Use the model for **relative** ranking of neighbourhoods rather than "
        "precise single-home valuation, given block-level aggregation and the price cap.",
        "3. For the highest-value coastal segments, supplement the linear model with "
        "local expertise, since the linear form understates coastal premiums.",
        "4. Revisit the model if new data lifts the value cap or adds property-level "
        "detail, which would materially improve top-of-market accuracy.",
        "",
    ]
    return "\n".join(lines)


def interpret_champion(*, top_n: int = 10, actor: str | None = None) -> dict[str, Any]:
    """
    Full explainability stage for the champion: table, chart, and briefing.

    Returns a summary dict with output paths and the top drivers.
    """

    ensure_dir(EXPLAINABILITY_DIR)
    bundle = load_champion_bundle()
    champion_label = f"{bundle['name']} {bundle['version']}"

    table = build_coefficient_table(bundle)
    table.attrs["champion_label"] = champion_label

    save_csv(table, COEFFICIENT_CSV)
    save_json(
        {
            "artifact": "coefficient_interpretation",
            "generated_at": utc_timestamp(),
            "champion": champion_label,
            "intercept": float(bundle["estimator"].intercept_),
            "coefficients": table.to_dict(orient="records"),
        },
        COEFFICIENT_JSON,
    )
    _top_drivers_chart(table, top_n, TOP_DRIVERS_PNG)

    briefing = _briefing_markdown(bundle, table, top_n=5)
    save_text(briefing, EXECUTIVE_BRIEFING_PATH)

    top5 = table.head(5)[["rank", "feature", "coefficient", "direction"]].to_dict(
        orient="records"
    )

    audit_logger.log_event(
        "explainability_generated",
        {
            "champion": champion_label,
            "top_drivers": [r["feature"] for r in top5],
            "briefing_path": str(EXECUTIVE_BRIEFING_PATH),
        },
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    lineage_tracker.record_node(
        "explainability",
        input_paths=[Path(bundle["registry_entry"]["model_path"])],
        output_paths=[COEFFICIENT_CSV, TOP_DRIVERS_PNG, EXECUTIVE_BRIEFING_PATH],
        script="src/explainability/interpretation.py",
        params={"champion": champion_label, "top_n": top_n},
    )

    logger.info(
        "Explainability complete for %s: top driver=%s -> %s",
        champion_label,
        top5[0]["feature"] if top5 else "n/a",
        EXECUTIVE_BRIEFING_PATH,
    )
    return {
        "champion": champion_label,
        "coefficient_csv": str(COEFFICIENT_CSV),
        "coefficient_json": str(COEFFICIENT_JSON),
        "top_drivers_png": str(TOP_DRIVERS_PNG),
        "executive_briefing": str(EXECUTIVE_BRIEFING_PATH),
        "top_drivers": top5,
    }
