"""
src.model_training.design_matrix
================================

PURPOSE
-------
Turn the governed, approved predictor columns of the engineered data into a
numeric modelling matrix, with every fitted statistic learned from the TRAINING
split only:

    numeric predictors : median-impute (train median) -> standardize (train mean/std)
    categorical (ocean_proximity) : one-hot (train categories, unknown -> all-zero)

The fitted :class:`ModelingPreprocessor` is persisted inside each model's joblib
bundle so evaluation and explainability transform validation/test identically,
and so the stored per-feature standard deviations let the explainability stage
translate standardized coefficients back to raw dollar effects.

PIPELINE POSITION
-----------------
    approval -> [design matrix] -> training / evaluation / explainability

DESIGN NOTE
-----------
This is a small, self-contained helper (not in the original file skeleton) that
exists so the leakage-safe transform is defined in exactly one importable place
and travels with the persisted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.constants import TARGET_COLUMN
from src.domain.schema import CATEGORICAL_COLUMNS


def _active_target() -> str:
    """Resolve the active project's target column (housing default if none).

    Only "there is no active project descriptor" is a legitimate reason to fall
    back. Catching every exception also swallowed real failures - unreadable
    YAML, a filesystem error, a malformed dataset.json - and silently returned
    the *housing* target for whatever project was actually selected. The
    resulting KeyError surfaced much later, somewhere unrelated.
    """

    try:
        from config.active import target_column

        return target_column()
    except (ImportError, AttributeError, FileNotFoundError, KeyError):
        # No descriptor for the active project: the documented fallback.
        return TARGET_COLUMN


def encode_target(y: pd.Series, positive_class: str | None) -> pd.Series:
    """
    Encode a BINARY classification target to 0/1 (positive class -> 1).

    When ``positive_class`` is None the series is returned unchanged: that covers
    regression, an already-numeric label, and multiclass targets (scikit-learn
    fits and scores string/multi-class labels natively, so no encoding is needed).
    """

    if positive_class is None:
        return y
    return (y.astype("string") == str(positive_class)).astype("int64")


def positive_class_index(estimator: Any) -> int:
    """
    Which ``predict_proba`` column holds the positive class.

    THE single definition of "positive" for a fitted classifier. Anything that
    pairs a probability with a truth indicator must derive both from here, or the
    two can disagree and every calibration number silently inverts.

    The rule is: prefer the literal label ``1`` wherever it appears, otherwise the
    last class. Note that this is NOT "the largest label" - for
    ``classes_ == [1, 2]`` it selects the column for ``1``, which is the *smaller*
    label. Reimplementing this as ``sorted(classes)[-1]`` therefore looks right and
    is wrong for any binary target not coded ``{0, 1}``.
    """

    classes = list(estimator.classes_)
    return classes.index(1) if 1 in classes else len(classes) - 1


def positive_class_label(estimator: Any) -> Any:
    """The label whose probability :func:`positive_proba` returns."""

    return list(estimator.classes_)[positive_class_index(estimator)]


def positive_proba(estimator: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Return P(positive class) for a fitted BINARY classifier."""

    return estimator.predict_proba(matrix)[:, positive_class_index(estimator)]


def class_scores(estimator: Any, matrix: pd.DataFrame) -> np.ndarray | None:
    """
    Probability scores shaped for the metric layer, or None if unavailable.

    Binary  -> 1-D positive-class probabilities.
    Multiclass -> the full (n_samples, n_classes) ``predict_proba`` matrix, whose
    columns are in ``estimator.classes_`` order as ROC-AUC one-vs-rest expects.
    """

    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(matrix)
    if proba.ndim > 1 and proba.shape[1] > 2:
        return proba
    return positive_proba(estimator, matrix)


@dataclass
class ModelingPreprocessor:
    """
    Leakage-safe numeric/categorical transformer fit on the training split.

    Attributes are populated by :meth:`fit` and are plain Python/NumPy so the
    object pickles cleanly with joblib.
    """

    predictor_columns: list[str]
    # Which predictor columns are nominal categoricals. When None the housing
    # schema's CATEGORICAL_COLUMNS is used (regression showcase, unchanged);
    # classification projects pass their configured categorical set.
    categorical_names: list[str] | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "ModelingPreprocessor":
        """Learn imputation medians, standardization stats, and categories from train."""

        cat_set = set(
            self.categorical_names
            if self.categorical_names is not None
            else CATEGORICAL_COLUMNS
        )
        present = [c for c in self.predictor_columns if c in train_df.columns]
        # A configured categorical is treated as categorical even if it stores
        # numeric codes (e.g. churn's SeniorCitizen 0/1); everything else that is
        # numeric is a numeric predictor.
        self.numeric_columns = [
            c
            for c in present
            if c not in cat_set and pd.api.types.is_numeric_dtype(train_df[c].dtype)
        ]
        self.categorical_columns = [c for c in present if c in cat_set]

        for column in self.numeric_columns:
            series = pd.to_numeric(train_df[column], errors="coerce")
            median = float(series.median())
            filled = series.fillna(median)
            std = float(filled.std(ddof=0))
            self.medians_[column] = median
            self.means_[column] = float(filled.mean())
            # Guard against zero-variance columns (avoid divide-by-zero).
            self.stds_[column] = std if std > 0 else 1.0

        feature_names: list[str] = list(self.numeric_columns)
        for column in self.categorical_columns:
            levels = sorted(
                train_df[column].astype("string").fillna("<MISSING>").unique().tolist()
            )
            self.categories_[column] = levels
            feature_names.extend(f"{column}_{level}" for level in levels)

        self.feature_names_ = feature_names
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric design matrix (standardized numerics + one-hot cats)."""

        if not self.fitted_:
            raise RuntimeError("ModelingPreprocessor.transform called before fit.")

        columns: dict[str, np.ndarray] = {}
        for column in self.numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce").fillna(
                self.medians_[column]
            )
            columns[column] = (
                (series - self.means_[column]) / self.stds_[column]
            ).to_numpy()

        for column in self.categorical_columns:
            values = df[column].astype("string").fillna("<MISSING>")
            for level in self.categories_[column]:
                columns[f"{column}_{level}"] = (values == level).astype("float64").to_numpy()

        return pd.DataFrame(columns, index=df.index)[self.feature_names_]

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on ``train_df`` then transform it."""

        return self.fit(train_df).transform(train_df)


def split_x_y(
    df: pd.DataFrame,
    predictor_columns: list[str],
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split ``df`` into (predictors, target).

    The predictor frame is restricted to ``predictor_columns`` (which must
    exclude the target), enforcing the trainer's guard that the target and any
    non-approved column never enter the model matrix. ``target_column`` defaults
    to the active project's target.
    """

    if target_column is None:
        target_column = _active_target()
    if target_column in predictor_columns:
        raise ValueError("Target column must not appear in predictor_columns.")
    missing = [c for c in predictor_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Approved predictor columns missing from data: {missing}")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' missing from data.")

    return df[predictor_columns].copy(), df[target_column].copy()
