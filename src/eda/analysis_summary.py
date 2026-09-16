"""
src.eda.analysis_summary
========================

PURPOSE
-------
Translate EDA outputs into an analytical notebook that explains, for the
training set only: which variables remain candidates and why, where
multicollinearity may exist, which relationships look nonlinear, which
deterministic features are worth creating, the likely modelling challenges, and
the recommended predictive/inferential workflow.

PIPELINE POSITION
-----------------
    EDA -> [analytical summary] -> feature engineering plan

Adapted from the professor's ``eda_analysis_summary_presentation.py``. The
analytical content is preserved: the marginal-association classifier, the
|r| >= 0.80 multicollinearity screen, a LOWESS curvature screen, the candidate
table, business interpretation, modelling challenges, and model-development
recommendations. Two deliberate deviations from the starter script: (1) LOWESS
is computed with a dependency-free binned-mean smoother instead of statsmodels
(statsmodels is not a project dependency); (2) the multi-hundred-line HTML/CSS
presentation layer is replaced by CSV + Markdown outputs, keeping the analysis
while dropping presentation chrome. Thresholds come from :mod:`config.constants`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.constants import (  # noqa: E402
    HIGH_PAIRWISE_CORRELATION,
    LOWESS_SAMPLE_SIZE,
    MATERIAL_OUTLIER_PERCENTAGE,
    MILD_CURVATURE_THRESHOLD,
    MODERATE_CURVATURE_THRESHOLD,
    RANDOM_STATE,
    STRONG_CURVATURE_THRESHOLD,
)
from config.paths import RESULTS_DIR, ensure_dir  # noqa: E402
from src.domain.schema import NUMERICAL_COLUMNS, SEMANTIC_SCHEMA, TARGET_COLUMN  # noqa: E402
from src.utils.file_utils import save_csv, save_text  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

SUMMARY_DIR: Path = RESULTS_DIR / "eda_analysis_summary"
FIGURES_DIR: Path = SUMMARY_DIR / "figures"


# ---------------------------------------------------------------------------
# Marginal association + multicollinearity screen
# ---------------------------------------------------------------------------

def classify_marginal_association(absolute_correlation: float) -> str:
    """Label the strength of a marginal linear association."""

    if not np.isfinite(absolute_correlation):
        return "Not applicable"
    if absolute_correlation >= 0.50:
        return "Strong marginal linear association"
    if absolute_correlation >= 0.20:
        return "Moderate marginal linear association"
    if absolute_correlation >= 0.10:
        return "Weak marginal linear association"
    return "Very weak marginal linear association"


def correlation_screen(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Predictor pairs with |r| >= HIGH_PAIRWISE_CORRELATION (redundancy risk)."""

    predictors = [c for c in correlation_matrix.columns if c != TARGET_COLUMN]
    rows: list[dict[str, object]] = []
    for i, first in enumerate(predictors):
        for second in predictors[i + 1:]:
            r = float(correlation_matrix.loc[first, second])
            if abs(r) < HIGH_PAIRWISE_CORRELATION:
                continue
            rows.append(
                {
                    "feature_1": first,
                    "feature_2": second,
                    "correlation": r,
                    "absolute_correlation": abs(r),
                    "eda_interpretation": "High pairwise association",
                    "machine_learning_implication": (
                        "Retain initially; predictive models may use both if "
                        "validation performance benefits."
                    ),
                    "inferential_implication": (
                        "Assess VIF, condition number, and theory before "
                        "coefficient interpretation."
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_1", "feature_2", "correlation", "absolute_correlation",
            "eda_interpretation", "machine_learning_implication",
            "inferential_implication",
        ],
    )


# ---------------------------------------------------------------------------
# Curvature (LOWESS) screen - dependency-free binned-mean smoother
# ---------------------------------------------------------------------------

def _binned_smoother(x: np.ndarray, y: np.ndarray, bins: int = 40) -> np.ndarray:
    """
    Nonparametric fit approximating LOWESS via binned means, interpolated back
    to every x. Used instead of statsmodels.lowess to avoid the dependency
    while preserving the "deviation from linearity" curvature signal.
    """

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    edges = np.linspace(x_sorted.min(), x_sorted.max(), bins + 1)
    centres, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x_sorted >= lo) & (x_sorted <= hi)
        if mask.any():
            centres.append((lo + hi) / 2.0)
            means.append(float(y_sorted[mask].mean()))
    if len(centres) < 2:
        return np.full_like(x, float(np.mean(y)))
    smoothed_sorted = np.interp(x_sorted, centres, means)
    smoothed = np.empty_like(smoothed_sorted)
    smoothed[order] = smoothed_sorted
    return smoothed


def curvature_score(feature: pd.Series, target: pd.Series) -> tuple[float, pd.DataFrame]:
    """RMS deviation of the nonparametric fit from a linear fit, scaled by std."""

    frame = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()

    if len(frame) < 50:
        return float("nan"), pd.DataFrame()
    if len(frame) > LOWESS_SAMPLE_SIZE:
        frame = frame.sample(n=LOWESS_SAMPLE_SIZE, random_state=RANDOM_STATE)

    frame = frame.sort_values("feature")
    x = frame["feature"].to_numpy(dtype="float64")
    y = frame["target"].to_numpy(dtype="float64")
    if np.isclose(np.std(x), 0):
        return 0.0, pd.DataFrame()

    slope, intercept = np.polyfit(x, y, deg=1)
    linear_fitted = intercept + slope * x
    smooth_fitted = _binned_smoother(x, y)

    target_std = float(np.std(y, ddof=1))
    score = (
        0.0
        if np.isclose(target_std, 0)
        else float(np.sqrt(np.mean((smooth_fitted - linear_fitted) ** 2)) / target_std)
    )
    plot_data = pd.DataFrame(
        {"feature": x, "target": y, "linear_fitted": linear_fitted, "smooth_fitted": smooth_fitted}
    )
    return score, plot_data


def classify_curvature(score: float) -> str:
    """Map a curvature score to a qualitative label."""

    if not np.isfinite(score):
        return "Insufficient data"
    if score < MILD_CURVATURE_THRESHOLD:
        return "Nearly linear"
    if score < MODERATE_CURVATURE_THRESHOLD:
        return "Mild curvature"
    if score < STRONG_CURVATURE_THRESHOLD:
        return "Moderate curvature"
    return "Strong nonlinear pattern"


def nonlinearity_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Provisional curvature screen with saved figures; not a final decision."""

    ensure_dir(FIGURES_DIR)
    rows: list[dict[str, object]] = []
    for feature in NUMERICAL_COLUMNS:
        if feature not in df.columns or feature == TARGET_COLUMN:
            continue
        score, plot_data = curvature_score(df[feature], df[TARGET_COLUMN])
        pattern = classify_curvature(score)
        figure_name = f"curvature_{feature}.png"
        rows.append(
            {
                "feature": feature,
                "curvature_deviation_score": score,
                "automated_screen": pattern,
                "analyst_decision": "Not reviewed",
                "figure_file": figure_name,
                "caution": (
                    "Provisional automated screen. Confirm with visual "
                    "inspection, specification tests, and validation."
                ),
                "exploratory_pattern": pattern,
            }
        )
        if plot_data.empty:
            continue
        plt.figure(figsize=(9, 6))
        plt.scatter(plot_data["feature"], plot_data["target"], alpha=0.15, s=10, label="Observations")
        plt.plot(plot_data["feature"], plot_data["linear_fitted"], linewidth=2, label="Linear fit")
        plt.plot(plot_data["feature"], plot_data["smooth_fitted"], linewidth=2, label="Smoothed")
        plt.xlabel(feature)
        plt.ylabel(TARGET_COLUMN)
        plt.title(f"Exploratory Linearity Review: {feature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / figure_name, dpi=200)
        plt.close()
    return (
        pd.DataFrame(rows)
        .sort_values("curvature_deviation_score", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Candidate explanatory variables
# ---------------------------------------------------------------------------

def _schema_reason(feature: str) -> str:
    semantic_type = str(SEMANTIC_SCHEMA.get(feature, {}).get("semantic_type", "predictor"))
    return {
        "spatial_coordinate": "Geographic position may capture regional market structure.",
        "count": "May contribute through scale, density, ratio, or interaction effects.",
        "nominal_categorical": "Systematic differences between categories may explain prices.",
        "discrete_numeric": "May have a nonlinear or piecewise relationship with house value.",
    }.get(semantic_type, "Remains conceptually relevant; assess jointly with other predictors.")


def candidate_variables(
    df: pd.DataFrame,
    target_correlations: pd.DataFrame,
    nonlinearity: pd.DataFrame,
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    """Table of candidate predictors with rationale and observed evidence."""

    corr_lookup = target_correlations.set_index("feature")["correlation_with_target"].to_dict()
    curve_lookup = nonlinearity.set_index("feature")["exploratory_pattern"].to_dict()
    outlier_lookup = (
        outliers.set_index("variable")["combined_flag_percentage"].to_dict()
        if not outliers.empty
        else {}
    )

    rows: list[dict[str, object]] = []
    for feature in df.columns:
        if feature == TARGET_COLUMN:
            continue
        r = float(corr_lookup.get(feature, np.nan))
        abs_r = abs(r) if np.isfinite(r) else np.nan
        rows.append(
            {
                "feature": feature,
                "semantic_type": SEMANTIC_SCHEMA.get(feature, {}).get("semantic_type", ""),
                "business_rationale": _schema_reason(feature),
                "correlation_with_target": r,
                "observed_linear_association": (
                    classify_marginal_association(abs_r)
                    if np.isfinite(abs_r)
                    else "Categorical predictor"
                ),
                "exploratory_nonlinearity": curve_lookup.get(feature, "Not applicable"),
                "potential_outlier_percentage": float(outlier_lookup.get(feature, 0.0)),
                "candidate_status": "RETAIN AS CANDIDATE",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modelling challenges + recommendations + conclusion
# ---------------------------------------------------------------------------

def modelling_challenges(
    df: pd.DataFrame,
    distribution_shapes: pd.DataFrame,
    screen: pd.DataFrame,
    nonlinearity: pd.DataFrame,
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    """Group likely modelling challenges into data / statistical / structural."""

    rows: list[dict[str, object]] = []
    target_row = distribution_shapes.loc[distribution_shapes["variable"] == TARGET_COLUMN]
    if not target_row.empty and abs(float(target_row.iloc[0]["skewness"])) >= 0.75:
        rows.append(
            {
                "challenge_group": "Statistical issue",
                "challenge": "Response asymmetry",
                "evidence": f"Response skewness = {float(target_row.iloc[0]['skewness']):.4f}.",
                "implication": "Consider transformation only if it improves validation adequacy.",
            }
        )
    if int(df.isna().sum().sum()) > 0:
        rows.append(
            {
                "challenge_group": "Data issue",
                "challenge": "Missing predictor values",
                "evidence": "Missing in: " + ", ".join(df.columns[df.isna().sum() > 0].tolist()),
                "implication": "Fit imputation using training data only.",
            }
        )
    if not screen.empty:
        rows.append(
            {
                "challenge_group": "Statistical issue",
                "challenge": "High predictor association",
                "evidence": f"{len(screen)} predictor pair(s) with |r| >= {HIGH_PAIRWISE_CORRELATION:.2f}.",
                "implication": "Run VIF/condition-number checks for inference; retain for ML.",
            }
        )
    curved = nonlinearity.loc[
        nonlinearity["exploratory_pattern"].isin(["Moderate curvature", "Strong nonlinear pattern"]),
        "feature",
    ].tolist()
    if curved:
        rows.append(
            {
                "challenge_group": "Statistical issue",
                "challenge": "Possible nonlinear relationships",
                "evidence": ", ".join(curved),
                "implication": "Compare linear, polynomial, and smooth specifications.",
            }
        )
    high_outlier = outliers.loc[outliers["combined_flag_percentage"] >= MATERIAL_OUTLIER_PERCENTAGE]
    if not high_outlier.empty:
        rows.append(
            {
                "challenge_group": "Data issue",
                "challenge": "Potential extreme observations",
                "evidence": ", ".join(high_outlier["variable"].tolist()),
                "implication": "Retain for prediction unless domain evidence supports removal.",
            }
        )
    rows.append(
        {
            "challenge_group": "Business / structural issue",
            "challenge": "Spatial heterogeneity",
            "evidence": "Coordinates exhibit joint geographic structure.",
            "implication": "Purely linear spatial effects may be insufficient.",
        }
    )
    return pd.DataFrame(rows)


def conclusion_text(screen: pd.DataFrame, nonlinearity: pd.DataFrame) -> str:
    """Concise executive EDA summary."""

    curved = nonlinearity.loc[
        nonlinearity["exploratory_pattern"].isin(["Moderate curvature", "Strong nonlinear pattern"]),
        "feature",
    ].tolist()
    return (
        "EXECUTIVE EDA SUMMARY\n"
        + "=" * 70 + "\n\n"
        + "- All schema-approved predictors remain candidates at this stage.\n"
        + "- Weak marginal correlation is not treated as evidence of unimportance.\n"
        + f"- {len(screen)} highly associated predictor pair(s) require VIF assessment.\n"
        + "- Provisional curvature screen flags: "
        + (", ".join(curved) if curved else "no predictors above the threshold") + ".\n"
        + "- Deterministic ratio features have direct business meaning and can be "
        + "generated automatically.\n"
        + "- EDA guides the next stage but does not determine the final model.\n"
        + "=" * 70 + "\n"
    )


def run_analysis_summary(
    train_df: pd.DataFrame,
    eda_outputs: dict[str, pd.DataFrame] | None = None,
    output_dir: Path = SUMMARY_DIR,
) -> dict[str, pd.DataFrame]:
    """Produce the analytical-summary tables and figures; return them."""

    ensure_dir(output_dir)
    ensure_dir(FIGURES_DIR)

    # Include the target so it appears in the correlation matrix / ranking.
    numeric = [c for c in NUMERICAL_COLUMNS + [TARGET_COLUMN] if c in train_df.columns]
    correlation_matrix = train_df[numeric].corr()

    if eda_outputs and "target_correlations" in eda_outputs:
        target_correlations = eda_outputs["target_correlations"]
    else:
        target_correlations = (
            correlation_matrix[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .rename("correlation_with_target")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    outliers = (
        eda_outputs.get("outlier_summary")
        if eda_outputs is not None
        else pd.DataFrame(columns=["variable", "combined_flag_percentage"])
    )
    shapes = (
        eda_outputs.get("distribution_shapes")
        if eda_outputs is not None
        else pd.DataFrame(columns=["variable", "skewness"])
    )

    screen = correlation_screen(correlation_matrix)
    nonlinearity = nonlinearity_screen(train_df)
    candidates = candidate_variables(train_df, target_correlations, nonlinearity, outliers)
    challenges = modelling_challenges(train_df, shapes, screen, nonlinearity, outliers)
    conclusion = conclusion_text(screen, nonlinearity)

    save_csv(candidates, output_dir / "candidate_explanatory_variables.csv")
    save_csv(screen, output_dir / "predictor_correlation_screen.csv")
    save_csv(nonlinearity, output_dir / "nonlinearity_screen.csv")
    save_csv(challenges, output_dir / "modelling_challenges.csv")
    save_text(conclusion, output_dir / "eda_conclusion.txt")
    logger.info("Analytical summary written to %s", output_dir)

    return {
        "candidates": candidates,
        "correlation_screen": screen,
        "nonlinearity": nonlinearity,
        "modelling_challenges": challenges,
    }
