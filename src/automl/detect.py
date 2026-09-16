"""
src.automl.detect
=================

PURPOSE
-------
Auto-understand an arbitrary uploaded CSV so the workbench can run its pipeline
on a dataset nobody hand-configured. Given a DataFrame it can:

- profile every column (dtype, missing %, cardinality, a few samples),
- guess the most likely target column,
- detect the learning task (regression vs binary classification),
- detect which predictors are categorical and which columns are identifiers to
  drop, and
- pick a sensible positive class for binary targets.

These are *heuristics with a human in the loop*: the GUI shows the guesses and
lets the user override the target and task before anything trains.

SCOPE
-----
Regression and classification (binary **and** multiclass). Binary targets get a
detected positive class and are scored on it; multiclass targets are left as
their native labels and scored one-vs-rest with macro averaging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Column-name tokens that strongly suggest a target/label.
_TARGET_NAME_HINTS = (
    "target", "label", "class", "outcome", "response", "churn", "default",
    "fraud", "converted", "survived", "y", "result", "status",
)
# Column-name tokens that suggest an identifier to drop from predictors.
_ID_NAME_HINTS = ("id", "identifier", "uuid", "index", "key")
# Object predictor columns with more distinct values than this are dropped
# (free text / high-cardinality codes would explode one-hot encoding).
_MAX_CATEGORICAL_CARDINALITY = 50


@dataclass
class DatasetDetection:
    """The auto-understanding result the GUI presents for confirmation."""

    target: str
    task: str  # "regression" | "classification"
    positive_class: str | None  # binary classification only; None otherwise
    categorical_columns: list[str]
    drop_columns: list[str]
    selection_metric: str
    n_classes: int | None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_multiclass(self) -> bool:
        """True for a classification target with more than two classes."""

        return self.task == "classification" and (self.n_classes or 0) > 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "task": self.task,
            "positive_class": self.positive_class,
            "categorical_columns": self.categorical_columns,
            "drop_columns": self.drop_columns,
            "selection_metric": self.selection_metric,
            "n_classes": self.n_classes,
            "warnings": self.warnings,
        }


def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns that are really numbers (e.g. ' 123.4', blanks)."""

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            # Treat as numeric only if almost everything converts (allow a few blanks).
            non_null = out[col].notna().sum()
            if non_null and converted.notna().sum() >= 0.95 * non_null:
                out[col] = converted
    return out


def profile_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column profile: dtype, missing %, distinct count, sample values."""

    n = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing_pct": round(100.0 * series.isna().mean(), 2) if n else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
                "sample": ", ".join(map(str, series.dropna().unique()[:4])),
            }
        )
    return rows


def _name_tokens(name: str) -> list[str]:
    """
    Split a column name into lowercase word tokens.

    Handles snake_case, kebab-case, spaces, and camelCase, so ``customerID`` ->
    ``["customer", "id"]`` while ``malic_acid`` -> ``["malic", "acid"]``. Token
    matching (not substring/suffix matching) is what keeps ordinary features
    ending in "id" — acid, valid, humid, paid — from being mistaken for
    identifiers and silently dropped.
    """

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name).strip())
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like row identifiers (all-unique, or id-named)."""

    n = len(df)
    ids: list[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        tokens = _name_tokens(col)
        nunique = df[col].nunique(dropna=True)
        looks_unique = n > 0 and nunique >= 0.99 * n and not pd.api.types.is_float_dtype(df[col])
        named_id = name in {"unnamed: 0", ""} or (
            any(t in _ID_NAME_HINTS for t in tokens)
            and not any(t in _TARGET_NAME_HINTS for t in tokens)
        )
        if looks_unique or named_id:
            ids.append(col)
    return ids


def guess_target(df: pd.DataFrame) -> str:
    """Best-guess target column: a name hint if present, else the last column."""

    lowered = {str(c).strip().lower(): c for c in df.columns}
    for hint in _TARGET_NAME_HINTS:
        if hint in lowered:
            return lowered[hint]
    # Otherwise a column whose name contains a hint token.
    for col in df.columns:
        name = str(col).strip().lower()
        if any(h in name for h in _TARGET_NAME_HINTS):
            return col
    return df.columns[-1]


def detect_task(series: pd.Series) -> tuple[str, int | None]:
    """
    Classify the target's learning task. Returns ``(task, n_classes)``.

    Non-numeric or low-cardinality-integer targets are classification (binary or
    multiclass, with ``n_classes`` recording which); everything else regression.
    """

    clean = series.dropna()
    n_unique = int(clean.nunique())

    numeric = pd.api.types.is_numeric_dtype(clean)
    integer_like = numeric and (clean == clean.round()).all()

    is_classification = (not numeric) or (integer_like and n_unique <= 15)
    if not is_classification:
        return "regression", None
    return "classification", n_unique


def guess_positive_class(series: pd.Series) -> str:
    """Pick the positive class of a binary target (Yes/True/1, else the minority)."""

    values = series.dropna()
    as_str = {str(v).strip().lower(): v for v in values.unique()}
    for token in ("yes", "true", "1", "churn", "positive", "y"):
        if token in as_str:
            return str(as_str[token])
    # Fall back to the minority class (churn/fraud/default are usually rarer).
    counts = values.value_counts()
    return str(counts.idxmin())


def detect_categoricals(df: pd.DataFrame, target: str, drop: list[str]) -> tuple[list[str], list[str]]:
    """
    Return ``(categorical_columns, extra_drops)``.

    Non-numeric predictors are categorical, unless their cardinality is so high
    they look like free text / codes, in which case they are dropped instead.
    """

    categoricals: list[str] = []
    extra_drops: list[str] = []
    excluded = set(drop) | {target}
    for col in df.columns:
        if col in excluded:
            continue
        # A column with no observed values at all carries no signal, and a
        # numeric one is actively harmful: its median/mean/std are all NaN, the
        # `std > 0` guard in the modelling preprocessor does not catch NaN
        # (NaN > 0 is False), and the NaN reaches sklearn as
        # "Input X contains NaN". Drop it here, as an unusable predictor.
        if df[col].nunique(dropna=True) == 0:
            extra_drops.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) > _MAX_CATEGORICAL_CARDINALITY:
            extra_drops.append(col)
        else:
            categoricals.append(col)
    return categoricals, extra_drops


def detect(df: pd.DataFrame, target: str | None = None) -> DatasetDetection:
    """Run the full auto-understanding and return a :class:`DatasetDetection`."""

    df = _coerce_numeric_like(df)
    warnings: list[str] = []

    if target is None:
        target = guess_target(df)
    if target not in df.columns:
        raise KeyError(f"Target '{target}' is not a column in the dataset.")

    task, n_classes = detect_task(df[target])

    drop = detect_id_columns(df)
    drop = [c for c in drop if c != target]
    categoricals, extra_drops = detect_categoricals(df, target, drop)
    drop = drop + extra_drops
    if extra_drops:
        warnings.append(
            f"Dropped high-cardinality text/code column(s): {', '.join(extra_drops)}."
        )
    if drop:
        warnings.append(f"Treating as identifiers (not predictors): {', '.join(drop)}.")

    positive_class = None
    if task == "classification":
        if n_classes == 1:
            warnings.append(
                f"Target '{target}' has only one distinct value — nothing to learn. "
                "Pick a different target."
            )
        elif n_classes == 2:
            positive_class = guess_positive_class(df[target])
        else:
            warnings.append(
                f"Target '{target}' has {n_classes} classes: training multiclass "
                "(one-vs-rest ROC-AUC, macro-averaged precision/recall/F1)."
            )

    selection_metric = "roc_auc" if task == "classification" else "rmse"

    return DatasetDetection(
        target=target,
        task=task,
        positive_class=positive_class,
        categorical_columns=categoricals,
        drop_columns=drop,
        selection_metric=selection_metric,
        n_classes=n_classes,
        warnings=warnings,
    )
