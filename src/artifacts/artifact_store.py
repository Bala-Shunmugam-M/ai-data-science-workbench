"""
src.artifacts.artifact_store
============================

PURPOSE
-------
Content-addressable file store. Any file can be copied into
``artifacts/store/`` under its sha256 content hash, with a sidecar metadata
record describing where it came from. Because the store key is the content hash,
identical files de-duplicate automatically and any stored artifact is verifiable
byte-for-byte.

PIPELINE POSITION
-----------------
Used by the lineage tracker and reporting to pin exact copies of key artifacts
(splits, engineered data, trained models, evaluation reports).

OUTPUTS
-------
    artifacts/store/<sha256>/<original_filename>
    artifacts/store/<sha256>/metadata.json
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from config.paths import ARTIFACT_STORE_DIR, PROJECT_ROOT, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.file_utils import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_file(
    source: Path,
    *,
    kind: str = "artifact",
    metadata: dict[str, Any] | None = None,
    store_dir: Path = ARTIFACT_STORE_DIR,
) -> dict[str, Any]:
    """
    Copy ``source`` into the content-hash store and return its store record.

    The returned record carries the content hash, the stored path, the original
    source path, size, kind, and a timestamp. Re-storing an unchanged file is a
    no-op copy (same hash directory).
    """

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Cannot store missing artifact: {source}")

    content_hash = _sha256(source)
    target_dir = ensure_dir(store_dir / content_hash)
    target_path = target_dir / source.name
    if not target_path.exists():
        shutil.copy2(source, target_path)

    try:
        source_display = str(source.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source_display = str(source)

    record: dict[str, Any] = {
        "kind": kind,
        "sha256": content_hash,
        "original_name": source.name,
        "source_path": source_display,
        "stored_path": str(target_path.relative_to(PROJECT_ROOT))
        if target_path.is_relative_to(PROJECT_ROOT)
        else str(target_path),
        "size_bytes": source.stat().st_size,
        "stored_at": utc_timestamp(),
        "metadata": metadata or {},
    }
    save_json(record, target_dir / "metadata.json")
    logger.info("Stored %s artifact %s -> %s", kind, source.name, content_hash[:12])
    return record
