"""
src.governance.lineage_tracker
==============================

PURPOSE
-------
Records end-to-end data and model lineage as a JSON graph:

    raw file -> processed -> splits -> engineered -> model version -> evaluation

Each node captures its input paths, output paths, the implementing script, a UTC
timestamp, and a hash of the parameters that produced it. This lets a reviewer
trace any evaluated model back to the exact raw data and transformations behind
it (the governance "lineage of one feature" demo requirement).

PIPELINE POSITION
-----------------
Nodes are appended by the preprocessing, feature, training, and evaluation
stages. The reporting stage counts nodes for the governance summary.

OUTPUTS
-------
    governance/lineage/lineage.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config.paths import LINEAGE_PATH, PROJECT_ROOT, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def file_sha256(path: Path) -> str | None:
    """Return the sha256 of a file's bytes, or ``None`` when it is absent."""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def params_hash(params: dict[str, Any] | None) -> str:
    """Return a stable sha256 over a canonical JSON encoding of ``params``."""

    payload = json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relativize(path: Path) -> str:
    """Render a path relative to the project root for portable lineage records."""

    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def record_node(
    stage: str,
    *,
    input_paths: list[Path] | None = None,
    output_paths: list[Path] | None = None,
    script: str = "",
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    path: Path = LINEAGE_PATH,
) -> dict[str, Any]:
    """
    Append (or replace, when ``stage`` repeats) one lineage node and persist it.

    Input/output files are hashed so the same run is reproducible and any change
    to an upstream file is detectable downstream.
    """

    inputs = input_paths or []
    outputs = output_paths or []

    node: dict[str, Any] = {
        "stage": stage,
        "timestamp": utc_timestamp(),
        "script": script,
        "inputs": [
            {"path": _relativize(p), "sha256": file_sha256(Path(p))} for p in inputs
        ],
        "outputs": [
            {"path": _relativize(p), "sha256": file_sha256(Path(p))} for p in outputs
        ],
        "params_hash": params_hash(params),
        "params": params or {},
    }
    if extra:
        node.update(extra)

    graph = _load(path)
    # A stage records a single canonical node; the latest run wins (idempotent).
    graph["nodes"] = [n for n in graph.get("nodes", []) if n.get("stage") != stage]
    graph["nodes"].append(node)
    graph["updated_at"] = utc_timestamp()
    _save(graph, path)

    logger.info("Lineage node '%s' recorded (%d node(s) total).", stage, len(graph["nodes"]))
    return node


def list_nodes(path: Path = LINEAGE_PATH) -> list[dict[str, Any]]:
    """Return all lineage nodes in insertion order."""

    return list(_load(path).get("nodes", []))


def summarize(path: Path = LINEAGE_PATH) -> dict[str, Any]:
    """Return a compact lineage summary (node count and stage order)."""

    nodes = list_nodes(path)
    return {"node_count": len(nodes), "stages": [n.get("stage") for n in nodes]}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"nodes": []}
    data = load_json(path)
    return data if isinstance(data, dict) else {"nodes": []}


def _save(graph: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    save_json(graph, path)
