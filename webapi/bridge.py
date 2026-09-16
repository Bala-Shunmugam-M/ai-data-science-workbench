"""
webapi.bridge
=============

PURPOSE
-------
CLI bridge a Node.js process spawns so the Next.js web app can drive the
existing workbench pipeline (``src.automl.detect`` / ``src.automl.register`` /
``src.pipelines.*`` / ``src.automl.report``) and stream real progress, without
modifying any of that existing code.

Two subcommands, both documented in ``docs/WEBAPP_PLAN.md`` sections 1.3/3/6:

    python webapi/bridge.py detect --csv <path> [--target <col>]
    python webapi/bridge.py analyze --csv <path> --name <display> \\
        --target <col> --task <regression|classification> [--positive-class <v>]

``detect`` prints exactly one JSON object to stdout. ``analyze`` prints one
JSON object per line (flushed immediately) as each stage starts/finishes/fails.
**stdout carries JSON and nothing else** -- every stray print, log line, and
warning from this process (ours or the pipeline's) is redirected to stderr, so
the Node side can parse stdout line-by-line without a sentinel or delimiter.

THE WORKBENCH_PROJECT ORDERING HAZARD
--------------------------------------
``config.paths`` reads ``os.environ["WORKBENCH_PROJECT"]`` exactly once, at
*module import time*, into module-level constants (``WORKSPACE_ROOT``,
``DATA_DIR``, ``ARTIFACTS_DIR``, ...). Once that module has been imported
anywhere in the process, setting the env var later has no effect -- Python
caches the module object and never re-executes it. ``src.automl.register`` (and
therefore anything that imports it) pulls in ``config.paths`` at its own module
level, so importing ``register_project`` before the slug is known and the env
var is set would permanently pin every pipeline stage to the wrong workspace.

The fix here: the project slug is computed locally (``_local_slugify``, a
deliberate small duplicate of ``src.automl.register.slugify`` -- see its
docstring for why it can't just call the real one), ``WORKBENCH_PROJECT`` is
set in ``os.environ`` from that slug, and *only then* are ``register_project``,
the pipeline modules, and ``src.automl.report`` imported (all inside the
``analyze`` code path, never at this file's top level). ``src.automl.detect``
has no ``config.paths`` dependency at all, so it is imported unconditionally at
the top of this file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Ensure the project root is importable when run as `python webapi/bridge.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `src.automl.detect` is pure pandas -- it never touches `config.paths` -- so
# it is always safe to import up front, regardless of subcommand.
from src.automl.detect import detect, guess_positive_class, profile_columns  # noqa: E402

# Captured before anything below can redirect it. All stdout output this
# process ever produces goes through `_REAL_STDOUT`, never the ambient
# `sys.stdout` (which `_stdout_silenced` swaps out during pipeline calls).
_REAL_STDOUT = sys.stdout

_RESERVED_SLUGS = {"california-housing", "california_housing", "churn"}


def _emit(event: dict[str, Any]) -> None:
    """Write one JSON object as a line to the real stdout, flushed immediately."""

    print(json.dumps(event), file=_REAL_STDOUT, flush=True)


@contextlib.contextmanager
def _stdout_silenced():
    """
    Swap `sys.stdout` for a throwaway buffer while pipeline code runs.

    ponytail: a blanket swap instead of auditing every transitive import
    (sklearn, matplotlib, pandas) for a stray print() -- cheap, and correct
    here because these stages are pure batch work with no real stdout use.
    Logging and warnings already default to stderr in this repo; this is the
    belt-and-suspenders guarantee for anything that doesn't.
    """

    real = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = real


def _local_slugify(display_name: str) -> str:
    """
    Duplicate of ``src.automl.register.slugify`` -- deliberately not imported.

    Importing ``src.automl.register`` imports ``config.paths`` as a side
    effect, which would freeze ``WORKSPACE_ROOT`` before we've set
    ``WORKBENCH_PROJECT``. This computes the same slug from the same
    dedup rule (checking ``workspaces/<slug>/dataset.json``) without pulling
    in that import chain, so the real ``register_project(..., slug=...)`` can
    be called with the env var already correct.
    """

    workspaces_dir = PROJECT_ROOT / "workspaces"
    slug = re.sub(r"[^a-z0-9]+", "-", display_name.strip().lower()).strip("-") or "dataset"
    slug = f"proj-{slug}" if slug in _RESERVED_SLUGS else slug
    base, i = slug, 2
    while (workspaces_dir / slug / "dataset.json").exists():
        slug = f"{base}-{i}"
        i += 1
    return slug


def _detection_json(d: Any) -> dict[str, Any]:
    """Map the snake_case `DatasetDetection` dataclass onto the camelCase wire shape."""

    return {
        "target": d.target,
        "task": d.task,
        "positiveClass": d.positive_class,
        "categoricalColumns": list(d.categorical_columns),
        "dropColumns": list(d.drop_columns),
        "selectionMetric": d.selection_metric,
        "nClasses": d.n_classes,
        "warnings": list(d.warnings),
    }


def _profile_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Map `profile_columns(df)` rows onto the ColumnProfile wire shape."""

    profile: list[dict[str, Any]] = []
    for row in profile_columns(df):
        col = row["column"]
        profile.append(
            {
                "name": col,
                "dtype": row["dtype"],
                "missing": int(df[col].isna().sum()),
                "missingPct": row["missing_pct"],
                "unique": row["n_unique"],
                "example": row["sample"],
            }
        )
    return profile


def _preview_json(df: pd.DataFrame, limit: int = 20) -> dict[str, Any]:
    """First `limit` rows as {columns, rows}, JSON-safe (NaN/NaT -> null)."""

    head = df.head(limit)
    # `to_json` does pandas' own numpy-scalar / NaN-to-null conversion instead
    # of hand-rolling one; parsing it back gives plain JSON-safe python values.
    rows = json.loads(head.to_json(orient="values", date_format="iso"))
    return {"columns": [str(c) for c in head.columns], "rows": rows}


def _read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Read a CSV that may not be UTF-8.

    Excel on a Western locale writes Windows-1252, so a single accented
    character makes a plain ``read_csv`` raise ``UnicodeDecodeError``. That is a
    routine upload, not a broken file, and it must not reach the user as a
    traceback. ``utf-8-sig`` first (strips the BOM Excel also likes to add),
    then cp1252, then latin-1 which cannot fail on any byte sequence.
    """

    last: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last = exc
    raise ValueError(
        f"Could not decode {Path(path).name} as text. Re-save it as UTF-8 CSV."
    ) from last


def _cmd_detect(args: argparse.Namespace) -> int:
    with _stdout_silenced():
        df = _read_csv_tolerant(args.csv)
        detection = detect(df, target=args.target)
        payload = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "detection": _detection_json(detection),
            "profile": _profile_json(df),
            "preview": _preview_json(df, limit=20),
        }
    _emit(payload)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    """Emit the human-approved semantic schema as a JSON table.

    The schema is a Python constant (``src.domain.schema.SEMANTIC_SCHEMA``), so
    the web app has no way to read it from disk. This is the only reason the
    subcommand exists: one row per column, in declaration order, with the rule
    fields the Data page shows.
    """

    with _stdout_silenced():
        from src.domain.schema import SEMANTIC_SCHEMA

        rows = [
            {
                "column": column,
                "role": rules.get("role"),
                "semantic_type": rules.get("semantic_type"),
                "expected_storage_type": rules.get("expected_storage_type"),
                "nullable": rules.get("nullable"),
                "minimum": rules.get("minimum"),
                "maximum": rules.get("maximum"),
                "whole_number": rules.get("whole_number"),
                "preprocessing_group": rules.get("preprocessing_group"),
                "missing_treatment": rules.get("missing_treatment"),
                "allowed_values": rules.get("allowed_values"),
            }
            for column, rules in SEMANTIC_SCHEMA.items()
        ]
    _emit({"columns": len(rows), "schema": rows})
    return 0


def _run_stage(stage_id: str, slug_holder: dict[str, str], fn: Callable[[], None]) -> bool:
    """Run one analyze stage, emitting started/done/failed events around it."""

    _emit({"stage": stage_id, "status": "started"})
    t0 = time.time()
    try:
        with _stdout_silenced():
            fn()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed.
        traceback.print_exc(file=sys.stderr)
        event: dict[str, Any] = {"stage": stage_id, "status": "failed", "message": str(exc)}
        if slug_holder.get("slug"):
            event["slug"] = slug_holder["slug"]
        _emit(event)
        return False
    event = {"stage": stage_id, "status": "done", "elapsedMs": int((time.time() - t0) * 1000)}
    if slug_holder.get("slug"):
        event["slug"] = slug_holder["slug"]
    _emit(event)
    return True


def _close_workspace_logging() -> None:
    """Detach every file log handler so a workspace directory can be removed.

    The workbench binds a ``FileHandler`` to
    ``<workspace>/artifacts/logs/workbench.log``. While it is attached, any
    later log record recreates the directory tree the moment after it is
    deleted - which is exactly what happened the first time this rollback was
    written.
    """

    import logging

    for logger_obj in [logging.getLogger()] + [
        logging.getLogger(name) for name in list(logging.root.manager.loggerDict)
    ]:
        for handler in list(getattr(logger_obj, "handlers", [])):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger_obj.removeHandler(handler)


def _discard_workspace(slug: str) -> None:
    """Delete a workspace registered moments ago by a run that then failed.

    Deliberately narrow: it refuses anything that is not a plain slug, and it
    only ever removes a directory directly under ``workspaces/``. The built-in
    California Housing project lives at the repo root rather than in
    ``workspaces/`` (``config/paths.py``), so it is unreachable from here by
    construction, and the two catalogue projects are named explicitly as a
    second guard.
    """

    import shutil

    if not slug or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        print(f"refusing to discard suspicious slug {slug!r}", file=sys.stderr)
        return
    if slug in {"churn", "california-housing", "california_housing"}:
        print(f"refusing to discard built-in project {slug!r}", file=sys.stderr)
        return

    workspaces_dir = (PROJECT_ROOT / "workspaces").resolve()
    target = (workspaces_dir / slug).resolve()
    if target.parent != workspaces_dir or not target.is_dir():
        return
    shutil.rmtree(target, ignore_errors=True)
    print(f"discarded incomplete workspace {slug!r}", file=sys.stderr)


def _cmd_analyze(args: argparse.Namespace) -> int:
    import os

    slug_holder: dict[str, str] = {}

    def extract() -> None:
        df = _read_csv_tolerant(args.csv)
        detection = detect(df, target=args.target)
        target = detection.target

        # Apply the caller's task/positive-class override, mirroring
        # app/pages/11_New_Project.py lines 76-89.
        if args.task == "regression":
            detection.task = "regression"
            detection.positive_class = None
            detection.n_classes = None
            detection.selection_metric = "rmse"
        elif args.task == "classification":
            if detection.task != "classification":
                detection.task = "classification"
                detection.selection_metric = "roc_auc"
                detection.n_classes = int(df[target].nunique(dropna=True))
                detection.positive_class = (
                    guess_positive_class(df[target]) if detection.n_classes == 2 else None
                )
            if args.positive_class:
                # Only meaningful for a binary target. Applying it to a
                # multiclass one silently binarises "chosen class vs the rest",
                # merging every other class together, while n_classes still
                # claims multiclass - so the metrics and the metadata describe
                # two different problems.
                if detection.n_classes not in (None, 2):
                    raise ValueError(
                        f"--positive-class is only valid for a binary target, but "
                        f"'{target}' has {detection.n_classes} classes. Drop the "
                        "flag to score all classes one-vs-rest, or pick a binary "
                        "target."
                    )
                if args.positive_class not in set(df[target].dropna().astype(str)):
                    present = sorted(set(df[target].dropna().astype(str)))[:10]
                    raise ValueError(
                        f"Positive class '{args.positive_class}' does not appear in "
                        f"'{target}'. Values present: {', '.join(present)}."
                    )
                detection.positive_class = args.positive_class
        else:
            raise ValueError(
                f"Unsupported task '{args.task}' (expected 'regression' or 'classification')."
            )

        # Slug must be known, and WORKBENCH_PROJECT set, before the first
        # import of anything that pulls in config.paths (see module docstring).
        slug = _local_slugify(args.name)
        os.environ["WORKBENCH_PROJECT"] = slug
        slug_holder["slug"] = slug

        from src.automl.register import register_project

        reg = register_project(args.name, df, detection, slug=slug)
        slug_holder["slug"] = reg["slug"]

        from src.pipelines import generic_pipeline

        generic_pipeline.run_prep()

    # The phases come from `generic_pipeline`, which is also what
    # `main.py autorun` runs. Keeping the call lists here instead let the two
    # paths drift: the web upload generated the driver table and the HTML
    # report while the CLI command did not.
    def analytics() -> None:
        from src.pipelines import generic_pipeline

        generic_pipeline.run_modelling()

    def insights() -> None:
        from src.pipelines import generic_pipeline

        generic_pipeline.run_insights()

    ok = _run_stage("extract", slug_holder, extract)
    if not ok:
        # `register_project` writes the workspace before prep can succeed, so a
        # failure here leaves a directory with a dataset.json, no results, and a
        # permanent dashboard entry reading "No results yet".
        #
        # The rollback happens *here*, after the stage has finished reporting,
        # rather than around `run_prep()`: the workspace logger is still bound
        # to that directory during the stage, and its next write recreates
        # `artifacts/logs/workbench.log` immediately after a deletion. Closing
        # the log handlers first is what makes the removal stick.
        _close_workspace_logging()
        _discard_workspace(slug_holder.get("slug", ""))
    if ok:
        ok = _run_stage("analytics", slug_holder, analytics)
    if ok:
        ok = _run_stage("insights", slug_holder, insights)
    return 0 if ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="webapi.bridge",
        description="Python bridge: detect a CSV's task/target, or run the full analyze pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_detect = subparsers.add_parser("detect", help="Profile a CSV and detect its task/target.")
    p_detect.add_argument("--csv", required=True, type=Path)
    p_detect.add_argument("--target", default=None)
    p_detect.set_defaults(handler=_cmd_detect)

    p_analyze = subparsers.add_parser(
        "analyze", help="Register the dataset and run prep -> train -> evaluate -> report."
    )
    p_analyze.add_argument("--csv", required=True, type=Path)
    p_analyze.add_argument("--name", required=True)
    p_analyze.add_argument("--target", required=True)
    p_analyze.add_argument("--task", required=True, choices=["regression", "classification"])
    p_analyze.add_argument("--positive-class", dest="positive_class", default=None)
    p_analyze.set_defaults(handler=_cmd_analyze)

    p_schema = subparsers.add_parser(
        "schema", help="Emit the semantic schema (a Python constant) as JSON."
    )
    p_schema.set_defaults(handler=_cmd_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001 - last-resort guard; nothing has hit stdout yet.
        traceback.print_exc(file=sys.stderr)
        print(f"bridge error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
