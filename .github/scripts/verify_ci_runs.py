"""
Verify that the CI pipeline runs actually produced results.

A pipeline can exit 0 and still have produced nothing useful: a stage can be
skipped, a champion can fail to be promoted, or a report can be written empty.
This script fails the build in those cases, so a green tick means the run
genuinely produced a champion, a model card and a report - not merely that no
exception escaped.

Run by: .github/workflows/ci.yml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACES = Path("workspaces")
EXPECTED_RUNS = 2  # one classification, one regression


def workspace_dirs() -> list[Path]:
    """Workspaces created by this CI run, newest first."""
    if not WORKSPACES.is_dir():
        return []
    candidates = [p for p in WORKSPACES.iterdir() if p.is_dir()]
    # The bridge slugifies the --name it was given, so both CI workspaces carry
    # "ci" as a slug segment. Matching on the segment rather than a substring
    # avoids picking up an unrelated workspace that merely contains those two
    # letters (for example "california-...").
    return sorted(
        (p for p in candidates if "ci" in p.name.lower().split("-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def main() -> int:
    runs = workspace_dirs()
    if len(runs) < EXPECTED_RUNS:
        present = (sorted(p.name for p in WORKSPACES.iterdir() if p.is_dir())
                   if WORKSPACES.is_dir() else [])
        print(f"FAIL: expected {EXPECTED_RUNS} CI workspaces, found "
              f"{len(runs)}. Workspaces present: {present}")
        return 1

    failures: list[str] = []
    for root in runs[:EXPECTED_RUNS]:
        selection = root / "artifacts" / "final_model_selection.json"
        card = root / "artifacts" / "trust" / "model_card.md"
        report = root / "artifacts" / "reports" / "project_report.html"

        if not selection.exists():
            failures.append(f"{root.name}: no final_model_selection.json")
            continue

        data = json.loads(selection.read_text(encoding="utf-8"))
        champion = data.get("champion_name")
        if not champion:
            failures.append(f"{root.name}: no champion was promoted")
        if not card.exists():
            failures.append(f"{root.name}: no model card")
        if not report.exists():
            failures.append(f"{root.name}: no HTML report")
        elif report.stat().st_size < 1024:
            failures.append(f"{root.name}: HTML report is suspiciously small "
                            f"({report.stat().st_size} bytes)")

        print(f"  {root.name}")
        print(f"    task      : {data.get('task')}")
        print(f"    champion  : {champion} {data.get('champion_version', '')}")
        print(f"    metric    : {data.get('selection_metric')}")
        print(f"    test      : {data.get('test_metrics')}")

    if failures:
        print("\nFAIL:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nOK: every CI run produced a champion, a model card and a report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
