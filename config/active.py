"""
config.active
=============

PURPOSE
-------
Resolve the *active project's* dataset configuration (task, target, metric,
categorical columns, split strategy) in one place, so downstream stages branch
on data-driven config instead of hard-coded regression globals.

The active project is selected by ``WORKBENCH_PROJECT`` (see :mod:`config.paths`).
Its slug maps to a ``datasets.yaml`` key: the housing showcase slug
``california-housing`` maps to the dataset key ``california_housing``; every
other project uses its slug as the dataset key directly.

PIPELINE POSITION
-----------------
Imported by the modelling matrix, trainer, evaluator, explainability, splitter,
proposal, and EDA stages to answer "what task/target/metric is this run?".
"""

from __future__ import annotations

from functools import lru_cache

from config.paths import ACTIVE_PROJECT
from src.data_manager.dataset_registry import DatasetDescriptor, get_dataset

#: Map a project slug (WORKBENCH_PROJECT) to its datasets.yaml key when they
#: differ. Only the housing showcase needs an alias.
_SLUG_TO_DATASET = {
    "california-housing": "california_housing",
    "": "california_housing",
}


def active_dataset_key(slug: str | None = None) -> str:
    """Return the datasets.yaml key for the active (or given) project slug."""

    slug = ACTIVE_PROJECT if slug is None else slug
    return _SLUG_TO_DATASET.get(slug, slug)


@lru_cache(maxsize=None)
def active_dataset(slug: str | None = None) -> DatasetDescriptor:
    """Return the :class:`DatasetDescriptor` for the active project."""

    return get_dataset(active_dataset_key(slug))


def task() -> str:
    """``"regression"`` or ``"classification"`` for the active project."""

    return active_dataset().task


def is_classification() -> bool:
    return task() == "classification"


def target_column() -> str:
    return active_dataset().target


def selection_metric() -> str:
    return active_dataset().selection_metric


def categorical_columns() -> list[str]:
    return list(active_dataset().categorical_columns)


def positive_class() -> str | None:
    return active_dataset().positive_class
