"""
deploy.huggingface.bootstrap
============================

PURPOSE
-------
Seed the demo projects at Docker image BUILD time, so a Space starts instantly
with something to browse instead of running a pipeline on a visitor's first
click.

The seeding itself lives in :mod:`src.automl.demo_seed`, which the Streamlit
entry point also calls. One implementation, two callers: keeping separate
copies is exactly how the CLI and web paths drifted apart earlier in this
project.

NOTE ON HOSTING
---------------
Hugging Face now requires a PRO subscription for Docker and Streamlit Spaces;
only static Spaces are free. This file is kept because it is correct and works
on a PRO account, but the free deployment path is Streamlit Community Cloud,
which has no build step and seeds through ``app/main.py`` instead.

USAGE
-----
    python deploy/huggingface/bootstrap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.automl.demo_seed import seed  # noqa: E402


def main() -> int:
    print("Seeding demo projects by running the real pipeline...")
    messages = seed()
    for message in messages:
        print(f"  {message}")

    # A Space whose dashboards are empty looks like broken software, so a total
    # failure must stop the image build rather than ship a hollow demo.
    if any(m.startswith("FAIL") for m in messages) and \
            not any(m.startswith("OK") for m in messages):
        print("ERROR: no demo project could be produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
