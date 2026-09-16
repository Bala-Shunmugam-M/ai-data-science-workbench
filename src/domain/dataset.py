"""
src.domain.dataset
==================

PURPOSE
-------
Lightweight ``Dataset`` value object bundling a dataset's name, DataFrame, and
the semantic schema that governs it. Passing this object between stages keeps
the name, data, and its contract travelling together.

PIPELINE POSITION
-----------------
Returned by the data manager after ingestion; consumed by downstream stages
that need both the frame and its schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.domain.schema import SEMANTIC_SCHEMA


@dataclass
class Dataset:
    """A named DataFrame paired with its governing semantic schema."""

    name: str
    df: pd.DataFrame
    schema: dict[str, dict[str, Any]] = field(default_factory=lambda: SEMANTIC_SCHEMA)

    @property
    def shape(self) -> tuple[int, int]:
        """Convenience passthrough to the underlying frame shape."""

        return self.df.shape

    def columns(self) -> list[str]:
        """Return the observed columns of the underlying frame."""

        return list(self.df.columns)

    def schema_columns(self) -> list[str]:
        """Return the columns declared in the semantic schema."""

        return list(self.schema.keys())
