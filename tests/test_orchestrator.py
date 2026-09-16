"""
tests.test_orchestrator
========================

Milestone-3 tests for the workflow orchestrator (:mod:`src.workflow.orchestrator`).

These tests exercise the orchestrator's own logic - dependency refusal, status
persistence, resume/force semantics, and range execution - against small
synthetic :class:`~src.workflow.pipeline_builder.StageSpec` registries built in
a ``tmp_path`` workspace. They never touch the real project artifacts or call
into the real pipeline stages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.workflow import orchestrator
from src.workflow.pipeline_builder import StageSpec


def _make_stage(
    name: str,
    tmp_path: Path,
    *,
    inputs: list[Path] | None = None,
    outputs: list[Path] | None = None,
    requires: list[str] | None = None,
    calls: list[str] | None = None,
    should_fail: bool = False,
) -> StageSpec:
    """Build a StageSpec whose ``run`` writes its outputs and records a call."""

    outputs = outputs or []

    def _run() -> None:
        if calls is not None:
            calls.append(name)
        if should_fail:
            raise RuntimeError(f"{name} exploded")
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"output of {name}", encoding="utf-8")

    return StageSpec(
        name=name,
        description=f"synthetic stage {name}",
        run=_run,
        inputs=inputs or [],
        outputs=outputs,
        requires=requires or [],
    )


# ---------------------------------------------------------------------------
# 1. Dependency refusal when declared inputs are missing.
# ---------------------------------------------------------------------------
def test_refuses_stage_with_missing_inputs(tmp_path):
    a_output = tmp_path / "a.txt"
    stage_a = _make_stage("a", tmp_path, outputs=[a_output])
    stage_b = _make_stage(
        "b", tmp_path, inputs=[a_output], outputs=[tmp_path / "b.txt"], requires=["a"]
    )

    status_path = tmp_path / "status.json"
    with pytest.raises(orchestrator.OrchestratorError) as excinfo:
        orchestrator.run_stage(
            "b", stages=[stage_a, stage_b], status_path=status_path
        )

    # The refusal message names the prerequisite stage.
    assert "a" in str(excinfo.value)
    assert "b" in str(excinfo.value)
    # No status was recorded for a stage that never ran.
    assert not status_path.exists() or "b" not in orchestrator.load_status(status_path)


def test_runs_once_dependency_is_satisfied(tmp_path):
    a_output = tmp_path / "a.txt"
    b_output = tmp_path / "b.txt"
    stage_a = _make_stage("a", tmp_path, outputs=[a_output])
    stage_b = _make_stage("b", tmp_path, inputs=[a_output], outputs=[b_output], requires=["a"])

    status_path = tmp_path / "status.json"
    orchestrator.run_stage("a", stages=[stage_a, stage_b], status_path=status_path)
    result = orchestrator.run_stage(
        "b", stages=[stage_a, stage_b], status_path=status_path
    )

    assert result["status"] == "ok"
    assert b_output.exists()


# ---------------------------------------------------------------------------
# 2. Status file is written with the documented shape.
# ---------------------------------------------------------------------------
def test_status_file_written_on_success(tmp_path):
    output = tmp_path / "out.txt"
    stage = _make_stage("solo", tmp_path, outputs=[output])
    status_path = tmp_path / "status.json"

    result = orchestrator.run_stage("solo", stages=[stage], status_path=status_path)

    assert status_path.exists()
    persisted = orchestrator.load_status(status_path)
    assert "solo" in persisted
    entry = persisted["solo"]
    assert entry["status"] == "ok"
    assert entry["duration_s"] >= 0.0
    assert "last_run" in entry and entry["last_run"]
    assert str(output) in entry["outputs"]
    assert result == {"stage": "solo", **entry}


def test_status_file_records_failure(tmp_path):
    stage = _make_stage("boom", tmp_path, should_fail=True)
    status_path = tmp_path / "status.json"

    with pytest.raises(RuntimeError, match="boom exploded"):
        orchestrator.run_stage("boom", stages=[stage], status_path=status_path)

    entry = orchestrator.load_status(status_path)["boom"]
    assert entry["status"] == "failed"
    assert "boom exploded" in entry["error"]


# ---------------------------------------------------------------------------
# 3. Resume (skip) vs. force-rerun.
# ---------------------------------------------------------------------------
def test_resume_skips_completed_stage_and_force_reruns(tmp_path):
    output = tmp_path / "out.txt"
    calls: list[str] = []
    stage = _make_stage("cached", tmp_path, outputs=[output], calls=calls)
    status_path = tmp_path / "status.json"

    first = orchestrator.run_stage("cached", stages=[stage], status_path=status_path)
    assert first["status"] == "ok"
    assert calls == ["cached"]

    # Second run without --force: outputs already exist -> skipped, run() not called again.
    second = orchestrator.run_stage("cached", stages=[stage], status_path=status_path)
    assert second["status"] == "skipped"
    assert calls == ["cached"]  # unchanged

    # --force reruns even though outputs already exist.
    third = orchestrator.run_stage(
        "cached", stages=[stage], status_path=status_path, force=True
    )
    assert third["status"] == "ok"
    assert calls == ["cached", "cached"]


# ---------------------------------------------------------------------------
# 4. Range execution via run_pipeline (--from/--to).
# ---------------------------------------------------------------------------
def test_run_pipeline_executes_declared_range_in_order(tmp_path):
    calls: list[str] = []
    a_out, b_out, c_out = (tmp_path / f"{n}.txt" for n in "abc")
    stage_a = _make_stage("a", tmp_path, outputs=[a_out], calls=calls)
    stage_b = _make_stage(
        "b", tmp_path, inputs=[a_out], outputs=[b_out], requires=["a"], calls=calls
    )
    stage_c = _make_stage(
        "c", tmp_path, inputs=[b_out], outputs=[c_out], requires=["b"], calls=calls
    )
    stages = [stage_a, stage_b, stage_c]
    status_path = tmp_path / "status.json"

    # Simulate that 'a' already ran in a prior process (its output exists on
    # disk) without recording status for it here.
    a_out.write_text("pre-existing", encoding="utf-8")

    results = orchestrator.run_pipeline(
        from_stage="b", to_stage="c", stages=stages, status_path=status_path
    )

    # Only 'b' and 'c' ran, in order; 'a' was never invoked (out of range).
    assert [r["stage"] for r in results] == ["b", "c"]
    assert calls == ["b", "c"]
    assert "a" not in orchestrator.load_status(status_path)


def test_run_pipeline_full_range_default(tmp_path):
    calls: list[str] = []
    a_out, b_out = tmp_path / "a.txt", tmp_path / "b.txt"
    stage_a = _make_stage("a", tmp_path, outputs=[a_out], calls=calls)
    stage_b = _make_stage(
        "b", tmp_path, inputs=[a_out], outputs=[b_out], requires=["a"], calls=calls
    )
    status_path = tmp_path / "status.json"

    results = orchestrator.run_pipeline(
        stages=[stage_a, stage_b], status_path=status_path
    )

    assert [r["stage"] for r in results] == ["a", "b"]
    assert all(r["status"] == "ok" for r in results)
    assert calls == ["a", "b"]


def test_run_pipeline_invalid_range_raises(tmp_path):
    stage_a = _make_stage("a", tmp_path, outputs=[tmp_path / "a.txt"])
    stage_b = _make_stage("b", tmp_path, inputs=[tmp_path / "a.txt"], requires=["a"])

    with pytest.raises(orchestrator.OrchestratorError):
        orchestrator.run_pipeline(
            from_stage="b",
            to_stage="a",
            stages=[stage_a, stage_b],
            status_path=tmp_path / "status.json",
        )
