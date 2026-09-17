"""
deploy.huggingface.bootstrap
============================

PURPOSE
-------
Seed the Hugging Face Space with two already-analysed demo projects, so a
visitor sees a working workbench on their first click rather than an empty
shell with nothing to browse.

WHY THIS IS NEEDED
------------------
The repository deliberately does not publish datasets or trained model
binaries: datasets are not ours to redistribute, and a .joblib is a pickle that
executes code when loaded. A fresh clone - which is exactly what the Space
builds from - therefore has the pipeline's CODE but none of its OUTPUT, so the
retention simulator would have no champion to load and the dashboards would be
blank.

This runs the real pipeline on the two 600-row samples that ARE committed, via
the same entry point the web console uses. Nothing is faked: the Space's demo
projects are genuine runs, and their model cards, fairness audits and
calibration figures are produced by the code being demonstrated.

WHEN IT RUNS
------------
At image BUILD time, not at startup. Baking the result into the image keeps the
Space's cold start fast and means a visitor never waits for a pipeline.

USAGE
-----
    python deploy/huggingface/bootstrap.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = PROJECT_ROOT / "webapp" / "public" / "samples"

# (sample file, display name, target column, task, positive class)
DEMOS = [
    ("telco-churn-sample.csv", "Telco Churn Demo", "Churn",
     "classification", "Yes"),
    ("california-housing-sample.csv", "California Housing Demo",
     "median_house_value", "regression", None),
]


def already_seeded() -> bool:
    """True when a demo workspace already carries a promoted champion."""
    workspaces = PROJECT_ROOT / "workspaces"
    if not workspaces.is_dir():
        return False
    for path in workspaces.glob("*demo*/artifacts/final_model_selection.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("champion_name"):
                return True
        except (OSError, ValueError):
            continue
    return False


def run_demo(sample: str, name: str, target: str, task: str,
             positive: str | None) -> bool:
    csv = SAMPLES / sample
    if not csv.exists():
        print(f"  SKIP {name}: {csv} is missing")
        return False

    command = [
        sys.executable, "webapi/bridge.py", "analyze",
        "--csv", str(csv),
        "--name", name,
        "--target", target,
        "--task", task,
    ]
    if positive:
        command += ["--positive-class", positive]

    print(f"  running: {name} ({task})")
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                            text=True)
    if result.returncode != 0:
        # The bridge writes JSON to stdout and diagnostics to stderr, so the
        # last stderr lines are the useful ones.
        tail = (result.stderr or "").strip().splitlines()[-3:]
        print(f"  FAILED {name}: {' | '.join(tail) or 'no output'}")
        return False
    print(f"  done: {name}")
    return True


def main() -> int:
    if already_seeded():
        print("Demo projects already present; nothing to do.")
        return 0

    print("Seeding demo projects by running the real pipeline...")
    ok = [run_demo(*demo) for demo in DEMOS]

    if not any(ok):
        # A Space with no browsable project is not worth shipping: the visitor
        # would land on an empty dashboard and conclude the tool is broken.
        print("ERROR: no demo project could be produced.")
        return 1

    if not all(ok):
        print("WARNING: some demos failed; the Space will show the rest.")

    print("\nWorkspaces now present:")
    for path in sorted((PROJECT_ROOT / "workspaces").glob("*")):
        if path.is_dir():
            champion = path / "artifacts" / "final_model_selection.json"
            mark = "champion" if champion.exists() else "no champion"
            print(f"  {path.name}  ({mark})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
