"""
src.preprocessing.encoder
=========================

PURPOSE
-------
One-hot encode ``ocean_proximity`` (the only categorical predictor) with the
encoder fit on the TRAINING split only.

PIPELINE POSITION
-----------------
    split -> imputer -> [encoder] -> scaler

LEAKAGE PREVENTION
------------------
Category levels are learned from train alone; ``handle_unknown='ignore'`` means
any category seen only in validation/test maps to all-zero indicators rather
than expanding the learned column space.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.domain.schema import CATEGORICAL_COLUMNS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OneHotCategoricalEncoder:
    """Thin wrapper around ``OneHotEncoder`` fit on training data only."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns: list[str] = columns or list(CATEGORICAL_COLUMNS)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> "OneHotCategoricalEncoder":
        """Learn category levels from the training split only."""

        self.encoder_.fit(train_df[self.columns].astype("string").fillna("<MISSING>"))
        self.feature_names_ = list(
            self.encoder_.get_feature_names_out(self.columns)
        )
        logger.info("Fitted one-hot encoder; features=%s", self.feature_names_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with the categorical columns replaced by indicators."""

        if not self.feature_names_:
            raise RuntimeError("Encoder.transform called before fit.")

        encoded = self.encoder_.transform(
            df[self.columns].astype("string").fillna("<MISSING>")
        )
        encoded_df = pd.DataFrame(
            encoded, columns=self.feature_names_, index=df.index
        )
        remainder = df.drop(columns=self.columns)
        return pd.concat([remainder, encoded_df], axis=1)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on ``train_df`` then transform it."""

        return self.fit(train_df).transform(train_df)
