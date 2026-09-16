"""
src.preprocessing.scaler
========================

PURPOSE
-------
Standardise numerical predictors with a ``StandardScaler`` fit on the TRAINING
split only. The target column is never scaled.

PIPELINE POSITION
-----------------
    split -> imputer -> encoder -> [scaler]

LEAKAGE PREVENTION
------------------
Means and standard deviations come from train alone and are reused unchanged on
validation and test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.domain.schema import NUMERICAL_COLUMNS, TARGET_COLUMN
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NumericStandardScaler:
    """Standard scaler over numerical predictors (target excluded)."""

    def __init__(self, columns: list[str] | None = None) -> None:
        # Scale numerical predictors only; never the target.
        self.columns: list[str] = columns or [
            column for column in NUMERICAL_COLUMNS if column != TARGET_COLUMN
        ]
        self.scaler_ = StandardScaler()
        self.fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "NumericStandardScaler":
        """Learn per-column mean/std from the training split only."""

        present = [c for c in self.columns if c in train_df.columns]
        self.columns = present
        self.scaler_.fit(train_df[present])
        self.fitted_ = True
        logger.info("Fitted standard scaler on %d numeric predictors", len(present))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with the numeric predictors standardised."""

        if not self.fitted_:
            raise RuntimeError("Scaler.transform called before fit.")

        result = df.copy()
        result[self.columns] = self.scaler_.transform(df[self.columns])
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on ``train_df`` then transform it."""

        return self.fit(train_df).transform(train_df)
