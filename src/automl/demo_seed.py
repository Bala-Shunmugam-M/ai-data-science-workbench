"""
src.automl.demo_seed
====================

PURPOSE
-------
Create demo projects by running the real pipeline on the sample data committed
to the repository.

WHY THIS EXISTS
---------------
The repository publishes the pipeline's CODE but not its OUTPUT: datasets are
not ours to redistribute, and a ``.joblib`` is a pickle that executes code when
loaded. Any deployment therefore starts from a clone that has empty dashboards
and a retention simulator with no champion to load - which reads as broken
software rather than as a missing dataset.

Seeding runs the genuine pipeline on the two 600-row samples that ARE
committed, through the same entry point the web console uses. Nothing is
fabricated: the demo projects' model cards, fairness audits and calibration
figures are produced by the code being demonstrated.

TWO CALLERS, ONE IMPLEMENTATION
-------------------------------
* ``app/main.py`` - seeds on first run, for hosts with no build step
  (Streamlit Community Cloud).
* ``deploy/huggingface/bootstrap.py`` - seeds at image build time, for hosts
  that have one.

They used to hold separate copies of this logic, which is precisely how the CLI
and web paths drifted apart earlier in this project's history.

PIPELINE POSITION
-----------------
    deployment start -> [demo_seed] -> a browsable workbench
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config.paths import PROJECT_ROOT

SAMPLES = PROJECT_ROOT / "webapp" / "public" / "samples"
WORKSPACES = PROJECT_ROOT / "workspaces"

# Seeding runs a full pipeline per demo. On a small free-tier container that is
# tens of seconds, so the timeout is generous but finite: a hung subprocess
# must not leave the app spinning forever on its first page load.
_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Demo:
    """One demo project to build from a committed sample."""

    sample: str
    name: str
    target: str
    task: str
    positive_class: str | None = None


DEMOS: tuple[Demo, ...] = (
    Demo("telco-churn-sample.csv", "Telco Churn Demo", "Churn",
         "classification", "Yes"),
    Demo("california-housing-sample.csv", "California Housing Demo",
         "median_house_value", "regression"),
)


def has_browsable_project() -> bool:
    """
    True when at least one workspace carries a promoted champion.

    This is the question that matters for a deployment: not "does a workspace
    directory exist" - the repository ships one for churn - but "is there a
    finished run a visitor can actually look at".
    """

    if not WORKSPACES.is_dir():
        return False
    for path in WORKSPACES.glob("*/artifacts/final_model_selection.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("champion_name"):
                return True
        except (OSError, ValueError):
            continue
    return False


def run_demo(demo: Demo) -> tuple[bool, str]:
    """Build one demo project. Returns ``(succeeded, message)``."""

    csv = SAMPLES / demo.sample
    if not csv.exists():
        return False, f"{demo.name}: sample missing at {csv}"

    command = [
        sys.executable, "webapi/bridge.py", "analyze",
        "--csv", str(csv),
        "--name", demo.name,
        "--target", demo.target,
        "--task", demo.task,
    ]
    if demo.positive_class:
        command += ["--positive-class", demo.positive_class]

    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                text=True, timeout=_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False, f"{demo.name}: timed out after {_TIMEOUT_SECONDS}s"

    if result.returncode != 0:
        # The bridge writes JSON to stdout and diagnostics to stderr, so the
        # last stderr lines carry the actual cause.
        tail = " | ".join((result.stderr or "").strip().splitlines()[-2:])
        return False, f"{demo.name}: {tail or 'failed with no output'}"

    return True, f"{demo.name}: built"


def seed(force: bool = False) -> list[str]:
    """
    Build the demo projects unless a browsable one already exists.

    Returns a list of human-readable status lines. Never raises: a deployment
    that cannot seed should still start and show what went wrong, because a
    stack trace on first page load tells a visitor nothing useful.
    """

    if not force and has_browsable_project():
        return ["Demo projects already present."]

    messages: list[str] = []
    for demo in DEMOS:
        ok, message = run_demo(demo)
        messages.append(("OK   " if ok else "FAIL ") + message)
    return messages


if __name__ == "__main__":
    for line in seed():
        print(line)
