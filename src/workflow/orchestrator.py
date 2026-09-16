"""
src.workflow.orchestrator
==========================

PURPOSE
-------
Light DAG runner over the stage registry declared in
:mod:`src.workflow.pipeline_builder`. It:

    1. Refuses to run a stage whose declared input files are missing,
       naming the prerequisite stage(s) that produce them.
    2. Persists per-stage status (``last_run`` UTC, ``duration_s``,
       ``status`` in {ok, failed, skipped}, ``outputs``) to
       ``artifacts/workflow_status.json``.
    3. Supports resuming (skip a stage whose declared outputs already exist)
       and forcing a full re-run.

It never reimplements stage logic - :class:`~src.workflow.pipeline_builder.StageSpec.run`
is always one of the existing ``src.pipelines.*`` (or governance) ``run()``
functions.

PIPELINE POSITION
-----------------
``main.py pipeline --from ... --to ... [--force]`` and ``main.py all`` are
both thin wrappers over :func:`run_pipeline`. The Streamlit GUI's Pipeline
page reads :func:`get_status` (read-only) and may shell out to
``python main.py pipeline ...`` to trigger a run.

OUTPUTS
-------
    artifacts/workflow_status.json
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from config.paths import WORKFLOW_STATUS_PATH, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.file_utils import load_json, save_json
from src.utils.logging_utils import get_logger
from src.workflow.pipeline_builder import StageSpec, build_stages

logger = get_logger(__name__)


class OrchestratorError(RuntimeError):
    """Raised when a stage cannot run: an unknown name or a missing input."""


# ---------------------------------------------------------------------------
# Status file I/O.
# ---------------------------------------------------------------------------
def load_status(path: Path = WORKFLOW_STATUS_PATH) -> dict[str, Any]:
    """Return the persisted per-stage status map (empty dict if none yet)."""

    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def save_status(status: dict[str, Any], path: Path = WORKFLOW_STATUS_PATH) -> Path:
    """Persist the per-stage status map."""

    ensure_dir(path.parent)
    return save_json(status, path)


def get_status(path: Path = WORKFLOW_STATUS_PATH) -> dict[str, Any]:
    """Read-only accessor for the GUI: the persisted per-stage status map."""

    return load_status(path)


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------
def _find(stages: list[StageSpec], name: str) -> StageSpec:
    for stage in stages:
        if stage.name == name:
            return stage
    names = [s.name for s in stages]
    raise OrchestratorError(f"Unknown stage '{name}'. Known stages: {names}.")


def _check_dependencies(spec: StageSpec) -> None:
    """Raise ``OrchestratorError`` naming the prerequisite stage(s) when a
    declared input file for ``spec`` is missing."""

    missing = [str(p) for p in spec.inputs if not Path(p).exists()]
    if not missing:
        return
    prereqs = ", ".join(spec.requires) if spec.requires else "an earlier stage"
    hint_from = spec.requires[0] if spec.requires else spec.name
    raise OrchestratorError(
        f"Cannot run stage '{spec.name}': missing input file(s) {missing}. "
        f"Run stage(s) [{prereqs}] first, e.g. "
        f"'python main.py pipeline --from {hint_from} --to {spec.name}'."
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def run_stage(
    name: str,
    *,
    force: bool = False,
    stages: list[StageSpec] | None = None,
    status_path: Path = WORKFLOW_STATUS_PATH,
) -> dict[str, Any]:
    """
    Run one stage by name.

    Dependency-checks first (always, even when the stage would be skipped),
    then either skips (outputs already present and ``force`` is False) or
    runs the stage, recording a status entry either way. Returns
    ``{"stage": name, **status_entry}``. Re-raises any exception from the
    stage after recording a ``failed`` status entry.
    """

    stages = stages if stages is not None else build_stages()
    spec = _find(stages, name)

    _check_dependencies(spec)

    status = load_status(status_path)

    if not force and spec.outputs and all(Path(p).exists() for p in spec.outputs):
        entry: dict[str, Any] = {
            "last_run": utc_timestamp(),
            "duration_s": 0.0,
            "status": "skipped",
            "outputs": [str(p) for p in spec.outputs],
        }
        status[name] = entry
        save_status(status, status_path)
        logger.info(
            "Stage '%s' skipped (outputs already present; use --force to rerun).",
            name,
        )
        return {"stage": name, **entry}

    logger.info("Running stage '%s': %s", name, spec.description)
    start = time.perf_counter()
    try:
        spec.run()
    except Exception as exc:  # noqa: BLE001 - record status, then propagate.
        duration = time.perf_counter() - start
        entry = {
            "last_run": utc_timestamp(),
            "duration_s": round(duration, 3),
            "status": "failed",
            "outputs": [],
            "error": str(exc),
        }
        status[name] = entry
        save_status(status, status_path)
        logger.error("Stage '%s' failed after %.2fs: %s", name, duration, exc)
        raise

    duration = time.perf_counter() - start
    entry = {
        "last_run": utc_timestamp(),
        "duration_s": round(duration, 3),
        "status": "ok",
        "outputs": [str(p) for p in spec.outputs if Path(p).exists()],
    }
    status[name] = entry
    save_status(status, status_path)
    logger.info("Stage '%s' completed in %.2fs.", name, duration)
    return {"stage": name, **entry}


def run_pipeline(
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
    stages: list[StageSpec] | None = None,
    status_path: Path = WORKFLOW_STATUS_PATH,
) -> list[dict[str, Any]]:
    """
    Run every stage in ``[from_stage, to_stage]`` (inclusive, registry order).

    ``from_stage``/``to_stage`` default to the first/last registered stage.
    Each stage is dependency-checked and status-tracked via :func:`run_stage`;
    a stage failure stops the run (the exception propagates after the
    ``failed`` status entry is written, so already-completed stages keep
    their recorded status).
    """

    stages = stages if stages is not None else build_stages()
    names = [s.name for s in stages]

    start_index = names.index(_find(stages, from_stage).name) if from_stage else 0
    end_index = names.index(_find(stages, to_stage).name) if to_stage else len(stages) - 1
    if start_index > end_index:
        raise OrchestratorError(
            f"--from '{from_stage}' comes after --to '{to_stage}' in the stage "
            f"order {names}."
        )

    results: list[dict[str, Any]] = []
    for name in names[start_index : end_index + 1]:
        results.append(
            run_stage(name, force=force, stages=stages, status_path=status_path)
        )
    return results
