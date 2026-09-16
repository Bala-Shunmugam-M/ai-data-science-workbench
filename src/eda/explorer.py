"""
src.eda.explorer
================

PURPOSE
-------
Exploratory data analysis on the TRAINING split only: descriptive statistics
(with skewness/kurtosis and distribution-shape labels), a missing-value report,
categorical frequency and categorical-vs-target summaries, a correlation matrix
and target-correlation ranking, a domain-anomaly report (households >
population), and an extreme-value screen combining IQR and modified Z-scores.
Matplotlib figures are saved (never shown).

PIPELINE POSITION
-----------------
    data understanding -> [EDA] -> analytical summary

Adapted from the professor's ``eda.py`` and ``revised_eda.py`` (merged into one
coherent module). Thresholds (IQR multiplier 1.5, modified-Z 3.5) come from
:mod:`config.constants`. No observations are removed, capped, or transformed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never displayed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.constants import IQR_MULTIPLIER, MODIFIED_Z_THRESHOLD  # noqa: E402
from config.paths import RESULTS_DIR, ensure_dir  # noqa: E402
from src.domain.schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.utils.data_utils import modified_z_scores  # noqa: E402
from src.utils.file_utils import save_csv, save_text  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

EDA_DIR: Path = RESULTS_DIR / "eda"
FIGURES_DIR: Path = EDA_DIR / "figures"


# ---------------------------------------------------------------------------
# Tabular analyses
# ---------------------------------------------------------------------------

# Numerical columns for statistics/correlation INCLUDE the target (the schema
# groups the target separately, but EDA correlates predictors against it).
NUMERIC_WITH_TARGET = NUMERICAL_COLUMNS + [TARGET_COLUMN]


def _present(columns: list[str], df: pd.DataFrame) -> list[str]:
    """The schema columns this frame actually has.

    The semantic schema describes the housing dataset. Indexing it into another
    project's frame raises ``KeyError: 'ocean_proximity'`` from inside a plotting
    loop, which reads as a broken stage rather than as "the schema does not
    describe this data". Every schema-driven loop filters through here; the callers
    that already inlined this filter are why the failure only showed up in some of
    them.
    """

    return [column for column in columns if column in df.columns]



def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numerical variables incl. skew/kurtosis."""

    numeric = [c for c in NUMERIC_WITH_TARGET if c in df.columns]
    summary = (
        df[numeric].describe().transpose().reset_index().rename(columns={"index": "variable"})
    )
    summary["missing_count"] = df[numeric].isna().sum().values
    summary["missing_percentage"] = (summary["missing_count"] / len(df)) * 100
    summary["skewness"] = df[numeric].skew(numeric_only=True).values
    summary["kurtosis"] = df[numeric].kurt(numeric_only=True).values
    return summary


def distribution_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution-shape labels per numeric variable (from revised_eda.py)."""

    rows: list[dict[str, object]] = []
    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce")
        skew = float(series.skew())
        if abs(skew) < 0.5:
            shape = "Approximately symmetric"
        elif skew > 0:
            shape = "Right-skewed"
        else:
            shape = "Left-skewed"
        rows.append(
            {
                "variable": column,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "standard_deviation": float(series.std(ddof=1)),
                "skewness": skew,
                "kurtosis": float(series.kurt()),
                "distribution_shape": shape,
            }
        )
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value counts per column, sorted descending."""

    missing_count = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percentage": (missing_count.values / len(df)) * 100,
        }
    )
    return report.sort_values("missing_count", ascending=False).reset_index(drop=True)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency distribution for categorical variables."""

    rows: list[dict[str, object]] = []
    for column in _present(CATEGORICAL_COLUMNS, df):
        counts = df[column].fillna("<MISSING>").value_counts(dropna=False)
        for category, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "category": category,
                    "count": int(count),
                    "percentage": (count / len(df)) * 100,
                }
            )
    return pd.DataFrame(rows)


def categorical_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Target mean/median/std by category level (drives categorical figures)."""

    frames: list[pd.DataFrame] = []
    for column in _present(CATEGORICAL_COLUMNS, df):
        grouped = (
            df.groupby(column, dropna=False, observed=True)[TARGET_COLUMN]
            .agg(
                observations="count",
                target_mean="mean",
                target_median="median",
                target_standard_deviation="std",
            )
            .reset_index()
        )
        grouped.insert(0, "categorical_variable", column)
        grouped = grouped.rename(columns={column: "category_level"})
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_reports(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix and a target-correlation ranking (abs-sorted)."""

    numeric = [c for c in NUMERIC_WITH_TARGET if c in df.columns]
    matrix = df[numeric].corr()
    target = (
        matrix[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target["absolute_correlation"] = target["correlation_with_target"].abs()
    return matrix, target


def domain_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where households exceed population (domain warning)."""

    anomalies = df.loc[df["households"] > df["population"]].copy()
    if not anomalies.empty:
        anomalies["population_per_household"] = anomalies["population"] / anomalies["households"]
    return anomalies


def outlier_reports(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IQR + modified-Z extreme-value screen (no data is modified)."""

    summary_rows: list[dict[str, object]] = []
    observation_frames: list[pd.DataFrame] = []

    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr

        iqr_flag = ((series < lower) | (series > upper)).fillna(False)
        mz = modified_z_scores(series)
        mz_flag = (mz.abs() > MODIFIED_Z_THRESHOLD).fillna(False)
        combined = iqr_flag | mz_flag
        valid_count = int(valid.count())

        summary_rows.append(
            {
                "variable": column,
                "observations": valid_count,
                "missing_count": int(series.isna().sum()),
                "q1": q1,
                "median": valid.median(),
                "q3": q3,
                "iqr": iqr,
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_flag_count": int(iqr_flag.sum()),
                "modified_z_flag_count": int(mz_flag.sum()),
                "combined_flag_count": int(combined.sum()),
                "combined_flag_percentage": (
                    int(combined.sum()) / valid_count * 100 if valid_count else 0.0
                ),
                "skewness": valid.skew(),
                "kurtosis": valid.kurt(),
            }
        )

        if int(combined.sum()) > 0:
            flagged = df.loc[combined].copy()
            flagged.insert(0, "outlier_variable", column)
            flagged.insert(1, "iqr_flag", iqr_flag.loc[combined].to_numpy())
            flagged.insert(2, "modified_z_flag", mz_flag.loc[combined].to_numpy())
            flagged.insert(3, "modified_z_score", mz.loc[combined].to_numpy())
            observation_frames.append(flagged)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("combined_flag_percentage", ascending=False)
        .reset_index(drop=True)
    )
    observations = (
        pd.concat(observation_frames, ignore_index=False) if observation_frames else pd.DataFrame()
    )
    return summary, observations


# ---------------------------------------------------------------------------
# Figures (savefig only)
# ---------------------------------------------------------------------------

def _save(figure_name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / figure_name, dpi=200)
    plt.close()


def render_figures(
    df: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    target_correlations: pd.DataFrame,
    categorical_target: pd.DataFrame,
) -> None:
    """Render and save the full EDA figure set."""

    ensure_dir(FIGURES_DIR)
    numeric = [c for c in NUMERICAL_COLUMNS if c in df.columns]

    # Target distribution.
    plt.figure(figsize=(9, 6))
    plt.hist(df[TARGET_COLUMN].dropna(), bins=50, edgecolor="black")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Frequency")
    plt.title("Distribution of Median House Value")
    _save("target_distribution.png")

    # Per-variable histograms and boxplots.
    for column in numeric:
        plt.figure(figsize=(9, 6))
        plt.hist(df[column].dropna(), bins=40, edgecolor="black")
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {column.replace('_', ' ').title()}")
        _save(f"{column}_distribution.png")

        plt.figure(figsize=(9, 4))
        plt.boxplot(df[column].dropna(), vert=False)
        plt.xlabel(column.replace("_", " ").title())
        plt.title(f"Boxplot of {column.replace('_', ' ').title()}")
        _save(f"{column}_boxplot.png")

    # Correlation heatmap.
    plt.figure(figsize=(11, 9))
    image = plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title("Correlation Matrix of Numerical Variables")
    _save("correlation_matrix.png")

    # Predictor-vs-target scatter plots.
    for column in [c for c in numeric if c != TARGET_COLUMN]:
        plot_data = df[[column, TARGET_COLUMN]].dropna()
        plt.figure(figsize=(9, 6))
        plt.scatter(plot_data[column], plot_data[TARGET_COLUMN], alpha=0.25, s=12)
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel(TARGET_COLUMN.replace("_", " ").title())
        plt.title(f"{column.replace('_', ' ').title()} and Median House Value")
        _save(f"{column}_vs_{TARGET_COLUMN}.png")

    # Categorical frequency and target-by-category.
    for column in _present(CATEGORICAL_COLUMNS, df):
        counts = df[column].fillna("<MISSING>").value_counts().sort_values(ascending=False)
        plt.figure(figsize=(9, 6))
        counts.plot(kind="bar")
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Number of observations")
        plt.title(f"Frequency Distribution of {column.replace('_', ' ').title()}")
        plt.xticks(rotation=45, ha="right")
        _save(f"{column}_frequency.png")

        categories = df[column].dropna().sort_values().unique()
        grouped_values = [df.loc[df[column] == c, TARGET_COLUMN].dropna() for c in categories]
        plt.figure(figsize=(10, 6))
        plt.boxplot(grouped_values, tick_labels=list(categories))
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel(TARGET_COLUMN.replace("_", " ").title())
        plt.title(f"Median House Value by {column.replace('_', ' ').title()}")
        plt.xticks(rotation=45, ha="right")
        _save(f"{TARGET_COLUMN}_by_{column}.png")

    # Geographic distribution coloured by target.
    geo = df[["longitude", "latitude", TARGET_COLUMN, "population"]].dropna()
    max_population = geo["population"].max()
    sizes = (geo["population"] / max_population) * 150
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        geo["longitude"], geo["latitude"], c=geo[TARGET_COLUMN],
        s=sizes.clip(lower=5), alpha=0.4,
    )
    plt.colorbar(scatter, label="Median house value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Distribution of House Values")
    _save("geographic_house_value_distribution.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_eda(train_df: pd.DataFrame, output_dir: Path = EDA_DIR) -> dict[str, pd.DataFrame]:
    """Compute EDA tables, save CSVs + figures + a text report, and return them."""

    ensure_dir(output_dir)
    ensure_dir(FIGURES_DIR)

    summary = summary_statistics(train_df)
    shapes = distribution_shapes(train_df)
    missing = missing_report(train_df)
    categorical = categorical_summary(train_df)
    categorical_target = categorical_target_summary(train_df)
    matrix, target_corr = correlation_reports(train_df)
    anomalies = domain_anomalies(train_df)
    outlier_summary, outlier_observations = outlier_reports(train_df)

    save_csv(summary, output_dir / "summary_statistics.csv")
    save_csv(shapes, output_dir / "numerical_distribution_summary.csv")
    save_csv(missing, output_dir / "missing_values.csv")
    save_csv(categorical, output_dir / "categorical_summary.csv")
    save_csv(categorical_target, output_dir / "categorical_target_summary.csv")
    save_csv(matrix, output_dir / "correlation_matrix.csv", index=True)
    save_csv(target_corr, output_dir / "target_correlations.csv")
    save_csv(anomalies, output_dir / "training_domain_anomalies.csv")
    save_csv(outlier_summary, output_dir / "outlier_summary.csv")
    save_csv(outlier_observations, output_dir / "outlier_observations.csv", index=True)

    render_figures(train_df, matrix, target_corr, categorical_target)

    strongest = target_corr.iloc[0]
    report = (
        "EXPLORATORY DATA ANALYSIS REPORT\n"
        + "=" * 60 + "\n\n"
        + "Dataset used: Training set only\n"
        + f"Observations: {len(train_df)}\n"
        + f"Columns: {train_df.shape[1]}\n"
        + f"Target variable: {TARGET_COLUMN}\n\n"
        + f"Target mean: {train_df[TARGET_COLUMN].mean():,.2f}\n"
        + f"Target median: {train_df[TARGET_COLUMN].median():,.2f}\n"
        + f"Target skewness: {train_df[TARGET_COLUMN].skew():.4f}\n"
        + f"Target kurtosis: {train_df[TARGET_COLUMN].kurt():.4f}\n\n"
        + f"Strongest target correlation: {strongest['feature']} "
        + f"({strongest['correlation_with_target']:.4f})\n"
        + f"Rows where households exceed population: {len(anomalies)}\n\n"
        + "No observations were removed, capped, winsorized, or transformed.\n"
        + "Any later decision must be documented and fitted on training data only.\n"
    )
    save_text(report, output_dir / "eda_report.txt")
    logger.info("EDA written to %s (figures in %s)", output_dir, FIGURES_DIR)

    return {
        "summary": summary,
        "distribution_shapes": shapes,
        "missing": missing,
        "categorical": categorical,
        "categorical_target": categorical_target,
        "correlation_matrix": matrix,
        "target_correlations": target_corr,
        "anomalies": anomalies,
        "outlier_summary": outlier_summary,
    }
