"""
src.artifacts.model_registry
============================

PURPOSE
-------
CRUD for ``models/model_registry.json`` - the catalogue of every trained model
version, its metrics, on-disk location, and champion status. The registry is the
single source of truth the evaluation stage reads to decide which models to
compare and which one to promote as champion.

PIPELINE POSITION
-----------------
Written by the training stage (``register``); read by evaluation and reporting.
The champion flag is set by the evaluation stage once, after the untouched-test
evaluation.

OUTPUTS
-------
    models/model_registry.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.paths import MODEL_REGISTRY_PATH, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"models": [], "champion": None, "updated_at": None}
    data = load_json(path)
    if not isinstance(data, dict):
        return {"models": [], "champion": None, "updated_at": None}
    data.setdefault("models", [])
    data.setdefault("champion", None)
    return data


def _save(registry: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    registry["updated_at"] = utc_timestamp()
    save_json(registry, path)


def register_model(
    entry: dict[str, Any],
    *,
    path: Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """
    Insert or replace a model version keyed by ``(name, version)``.

    ``entry`` must carry at least ``name`` and ``version``; callers include
    metrics, ``model_path``, ``metadata_path``, ``data_hash`` and ``params``.
    """

    if "name" not in entry or "version" not in entry:
        raise ValueError("Registry entry requires 'name' and 'version'.")

    registry = _load(path)
    key = (entry["name"], entry["version"])
    registry["models"] = [
        m for m in registry["models"] if (m.get("name"), m.get("version")) != key
    ]
    stamped = {**entry, "registered_at": utc_timestamp()}
    registry["models"].append(stamped)
    _save(registry, path)
    logger.info("Registered model %s %s.", entry["name"], entry["version"])
    return stamped


def list_models(path: Path = MODEL_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Return every registered model version."""

    return list(_load(path).get("models", []))


def get_model(
    name: str,
    version: str | None = None,
    *,
    path: Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any] | None:
    """Return a specific model version, or the latest version when ``version`` is None."""

    matches = [m for m in list_models(path) if m.get("name") == name]
    if version is not None:
        matches = [m for m in matches if m.get("version") == version]
    if not matches:
        return None
    return sorted(matches, key=lambda m: m.get("version", ""))[-1]


def next_version(name: str, *, path: Path = MODEL_REGISTRY_PATH) -> str:
    """Return the next ``vNNN`` version string for ``name`` (v001 if new)."""

    versions = [
        m.get("version", "")
        for m in list_models(path)
        if m.get("name") == name
    ]
    numbers = [int(v[1:]) for v in versions if v.startswith("v") and v[1:].isdigit()]
    return f"v{(max(numbers) + 1) if numbers else 1:03d}"


def set_champion(
    name: str,
    version: str,
    *,
    path: Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """Mark one model version as the registry champion (clears any previous)."""

    registry = _load(path)
    found = False
    for model in registry["models"]:
        is_champ = model.get("name") == name and model.get("version") == version
        model["is_champion"] = is_champ
        found = found or is_champ
    if not found:
        raise KeyError(f"Cannot promote unknown model: {name} {version}")
    registry["champion"] = {"name": name, "version": version}
    _save(registry, path)
    logger.info("Champion set to %s %s.", name, version)
    return registry["champion"]


def get_champion(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any] | None:
    """Return the champion's full registry entry, or ``None`` when unset."""

    registry = _load(path)
    champion = registry.get("champion")
    if not champion:
        return None
    return get_model(champion["name"], champion["version"], path=path)
