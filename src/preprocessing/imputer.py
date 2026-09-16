"""
src.preprocessing.imputer
=========================

PURPOSE
-------
Median imputation for ``total_bedrooms`` fit on the TRAINING split only and
applied unchanged to validation and test. The fitted median is persisted to
``artifacts/`` so the transform is auditable and reproducible.

PIPELINE POSITION
-----------------
    split -> [imputer] -> encoder -> scaler

LEAKAGE PREVENTION
------------------
The median is computed from train alone; validation/test never contribute to
the statistic. The schema marks ``total_bedrooms`` as
``median_imputation_after_split`` for exactly this reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import ARTIFACTS_DIR
from src.utils.file_utils import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

IMPUTE_COLUMN = "total_bedrooms"
ARTIFACT_PATH: Path = ARTIFACTS_DIR / "preprocessing" / "imputer_medians.json"


class MedianImputer:
    """Median imputer for a set of columns, fit on training data only."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns: list[str] = columns or [IMPUTE_COLUMN]
        self.medians_: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "MedianImputer":
        """Learn the per-column median from the training split only."""

        for column in self.columns:
            if column in train_df.columns:
                self.medians_[column] = float(
                    pd.to_numeric(train_df[column], errors="coerce").median()
                )
        logger.info("Fitted median imputer on train: %s", self.medians_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the train-derived medians."""

        if not self.medians_:
            raise RuntimeError("MedianImputer.transform called before fit.")

        result = df.copy()
        for column, median in self.medians_.items():
            if column in result.columns:
                result[column] = pd.to_numeric(
                    result[column], errors="coerce"
                ).fillna(median)
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on ``train_df`` then transform it."""

        return self.fit(train_df).transform(train_df)

    def save(self, path: Path = ARTIFACT_PATH) -> Path:
        """Persist the fitted medians for auditing/reuse."""

        return save_json({"medians": self.medians_}, path)
