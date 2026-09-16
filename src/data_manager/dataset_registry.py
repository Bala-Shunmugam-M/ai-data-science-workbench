"""
src.data_manager.dataset_registry
=================================

PURPOSE
-------
Read the dataset registry from ``config/datasets.yaml`` and expose typed
descriptors. Keeps the platform dataset-agnostic: adding a project is a YAML
edit, not a code change.

PIPELINE POSITION
-----------------
Consulted by the data manager to resolve a dataset name into its Kaggle id,
file name, and target column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from config.paths import CONFIG_DIR, PROJECT_ROOT
from src.utils.common import load_yaml

DATASETS_YAML = CONFIG_DIR / "datasets.yaml"
#: Uploaded ("bring your own dataset") projects each drop a descriptor here.
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"


@dataclass(frozen=True)
class DatasetDescriptor:
    """Typed view of one entry in ``datasets.yaml``."""

    name: str
    display_name: str
    source: str
    kaggle_id: str
    file: str
    target: str
    description: str = ""
    # Task-aware fields (default to the regression showcase's behaviour so an
    # older entry that omits them keeps working unchanged).
    task: str = "regression"
    stratify: str = "income"
    selection_metric: str = "rmse"
    positive_class: str | None = None
    categorical_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    monthly_revenue_column: str | None = None


def _descriptor_from_spec(name: str, spec: dict) -> DatasetDescriptor:
    """Build a typed descriptor from a raw spec dict (datasets.yaml or dataset.json)."""

    return DatasetDescriptor(
        name=name,
        display_name=str(spec.get("display_name", name)),
        source=str(spec.get("source", "")),
        kaggle_id=str(spec.get("kaggle_id", "")),
        file=str(spec.get("file", "")),
        target=str(spec.get("target", "")),
        description=str(spec.get("description", "")),
        # Uploaded datasets default to no stratification; registered ones say so.
        task=str(spec.get("task", "regression")),
        stratify=str(spec.get("stratify", "income")),
        selection_metric=str(spec.get("selection_metric", "rmse")),
        positive_class=(
            None if spec.get("positive_class") is None else str(spec.get("positive_class"))
        ),
        categorical_columns=tuple(spec.get("categorical_columns", []) or []),
        drop_columns=tuple(spec.get("drop_columns", []) or []),
        monthly_revenue_column=(
            None if spec.get("monthly_revenue_column") is None
            else str(spec.get("monthly_revenue_column"))
        ),
    )


def load_dynamic_projects() -> dict[str, DatasetDescriptor]:
    """Discover uploaded projects from ``workspaces/<slug>/dataset.json``."""

    registry: dict[str, DatasetDescriptor] = {}
    if not WORKSPACES_DIR.exists():
        return registry
    for descriptor_path in sorted(WORKSPACES_DIR.glob("*/dataset.json")):
        try:
            spec = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = spec.get("name") or descriptor_path.parent.name
        registry[name] = _descriptor_from_spec(name, spec)
    return registry


def load_registry() -> dict[str, DatasetDescriptor]:
    """Load all registered datasets: the built-in datasets.yaml plus uploads."""

    raw = load_yaml(DATASETS_YAML)
    entries = raw.get("datasets", {}) or {}

    registry: dict[str, DatasetDescriptor] = {
        name: _descriptor_from_spec(name, spec) for name, spec in entries.items()
    }
    # Uploaded projects override nothing built-in; they add new keys.
    registry.update(load_dynamic_projects())
    return registry


def get_dataset(name: str) -> DatasetDescriptor:
    """Return the descriptor for ``name`` or raise a clear KeyError."""

    registry = load_registry()
    if name not in registry:
        available = ", ".join(sorted(registry)) or "(none registered)"
        raise KeyError(
            f"Dataset '{name}' is not registered. Available: {available}"
        )
    return registry[name]
