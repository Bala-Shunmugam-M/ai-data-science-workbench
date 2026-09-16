"""
src.domain.data_profile
========================

PURPOSE
-------
Dataset-agnostic *structural* profiler. It reports observed structure only
(dtypes, missingness, cardinality, numeric ranges, candidate flags) and makes
no semantic assumptions.

PIPELINE POSITION
-----------------
    ingestion -> [data_profile] -> semantic validation -> preprocessing

Adapted from the professor's ``data_profile.py``: the ``infer_type``,
``profile_column``, and low-cardinality candidate logic are preserved; the CLI
``main`` was refactored into ``profile_dataframe`` / ``write_profile`` so the
platform can call it programmatically and still emit the same CSV/HTML/TXT
outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import RESULTS_DIR, ensure_dir
from src.utils.data_utils import pct
from src.utils.file_utils import save_csv, save_text

PROFILE_DIR: Path = RESULTS_DIR / "data_profile"


def looks_datetime(series: pd.Series) -> bool:
    """Heuristic: does a string column parse as datetime for >= 90% of values?"""

    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        return False

    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False

    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.90


def infer_type(series: pd.Series) -> str:
    """Infer a coarse structural type label without semantic assumptions."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "Datetime"
    if looks_datetime(series):
        return "Datetime candidate"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "Integer numeric"
    if pd.api.types.is_float_dtype(series.dtype):
        non_missing = series.dropna()
        if not non_missing.empty and np.allclose(non_missing, np.round(non_missing)):
            return "Integer-valued numeric"
        return "Continuous numeric"
    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "String / categorical candidate"
    return "Other"


def profile_column(
    name: str,
    series: pd.Series,
    row_count: int,
    position: int,
) -> dict[str, object]:
    """Produce one structural-profile row for a single column."""

    non_missing = series.dropna()
    unique_count = int(non_missing.nunique())
    unique_ratio = unique_count / len(non_missing) if len(non_missing) else 0.0

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_non_missing = numeric.dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series.dtype)

    def _numeric_stat(func) -> float:
        if is_numeric and not numeric_non_missing.empty:
            return float(func(numeric_non_missing))
        return np.nan

    return {
        "column_position": position,
        "column_name": name,
        "observed_pandas_dtype": str(series.dtype),
        "inferred_structural_type": infer_type(series),
        "observations": row_count,
        "non_missing_count": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": pct(int(series.isna().sum()), row_count),
        "unique_count": unique_count,
        "unique_percentage": pct(unique_count, row_count),
        "binary_candidate": unique_count == 2,
        "low_cardinality_candidate": (
            unique_count > 0 and (unique_count <= 20 or unique_ratio <= 0.05)
        ),
        "identifier_candidate": (len(non_missing) > 0 and unique_ratio >= 0.98),
        "integer_valued_candidate": (
            bool(np.allclose(numeric_non_missing, np.round(numeric_non_missing)))
            if is_numeric and not numeric_non_missing.empty
            else False
        ),
        "minimum": _numeric_stat(lambda s: s.min()),
        "maximum": _numeric_stat(lambda s: s.max()),
        "mean": _numeric_stat(lambda s: s.mean()),
        "median": _numeric_stat(lambda s: s.median()),
        "standard_deviation": _numeric_stat(lambda s: s.std(ddof=1)),
        "zero_count": (
            int((numeric_non_missing == 0).sum()) if is_numeric else np.nan
        ),
        "negative_count": (
            int((numeric_non_missing < 0).sum()) if is_numeric else np.nan
        ),
        "example_values": " | ".join(
            non_missing.astype(str).drop_duplicates().head(5).tolist()
        ),
    }


def profile_dataframe(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (column profile, low-cardinality value counts) for ``data``."""

    profile = pd.DataFrame(
        [
            profile_column(column, data[column], len(data), position)
            for position, column in enumerate(data.columns, start=1)
        ]
    )

    candidate_rows: list[dict[str, object]] = []
    for column in profile.loc[profile["low_cardinality_candidate"], "column_name"]:
        counts = (
            data[column].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        )
        for value, count in counts.items():
            candidate_rows.append(
                {
                    "column_name": column,
                    "observed_value": str(value),
                    "frequency": int(count),
                    "percentage": pct(int(count), len(data)),
                }
            )

    return profile, pd.DataFrame(candidate_rows)


def write_profile(
    data: pd.DataFrame,
    source_label: str,
    output_dir: Path = PROFILE_DIR,
) -> pd.DataFrame:
    """Profile ``data`` and write CSV + summary outputs; return the profile."""

    ensure_dir(output_dir)
    profile, candidates = profile_dataframe(data)

    save_csv(profile, output_dir / "data_profile.csv")
    save_csv(candidates, output_dir / "categorical_candidates.csv")

    summary = (
        "STRUCTURAL DATA PROFILE\n"
        f"Input: {source_label}\n"
        f"Rows: {len(data)}\n"
        f"Columns: {len(data.columns)}\n"
        f"Missing values: {int(data.isna().sum().sum())}\n"
        f"Duplicate rows: {int(data.duplicated().sum())}\n"
        "No semantic assumptions were applied.\n"
    )
    save_text(summary, output_dir / "profile_summary.txt")

    return profile
