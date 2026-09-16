"""
main.py
=======

PURPOSE
-------
Command-line entry point for the AI Data Science Workbench.

Each subcommand runs one pipeline stage and writes its outputs under ``data/``
and ``results/`` (and, for feature engineering, ``governance/approvals/``):

    python main.py ingest       # download/cache -> profile -> validate
    python main.py validate     # (re)run semantic-schema validation
    python main.py preprocess   # clean -> split -> fit train-only transforms
    python main.py split        # split only (requires processed dataset)
    python main.py understand   # data-understanding report on train
    python main.py eda          # EDA + analytical summary on train
    python main.py features     # feature proposal + execute automatic features
    python main.py propose      # model proposal (governed candidate list)
    python main.py approve      # approve features and/or models (gates training)
    python main.py train        # train every approved model
    python main.py evaluate     # compare, pick champion, test-set evaluation
    python main.py explain      # champion coefficients + executive briefing
    python main.py trust        # subgroup fairness, permutation importance,
                                #   calibration, model card
    python main.py report       # self-contained HTML project report
    python main.py all          # run the full chain end to end (always reruns)
    python main.py pipeline     # orchestrated run: --from/--to/--force (resumable)

``all`` and ``pipeline`` both delegate to the Milestone-3 workflow
orchestrator (:mod:`src.workflow.orchestrator`), which dependency-checks each
stage's declared inputs and persists per-stage status to
``artifacts/workflow_status.json``. ``all`` always reruns every stage (its
prior end-to-end behavior is unchanged); ``pipeline`` defaults to resuming
(skipping a stage whose declared outputs already exist) unless ``--force`` is
given. The Streamlit GUI (``streamlit run app/main.py``) is a read-only
dashboard over the same generated artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable when run as ``python main.py``.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.governance import approvals  # noqa: E402
from src.pipelines import (  # noqa: E402
    churn_pipeline,
    eda_pipeline,
    evaluation_pipeline,
    feature_pipeline,
    generic_pipeline,
    ingestion_pipeline,
    preprocessing_pipeline,
    proposal_pipeline,
    reporting_pipeline,
    training_pipeline,
    trust_pipeline,
)
from src.utils.logging_utils import get_logger  # noqa: E402
from src.workflow import orchestrator, pipeline_builder  # noqa: E402

logger = get_logger("workbench.cli")


def _cmd_ingest(args: argparse.Namespace) -> None:
    ingestion_pipeline.run(args.dataset, prefer_cache=not args.download)


def _cmd_validate(args: argparse.Namespace) -> None:
    ingestion_pipeline.run_validate(args.dataset)


def _cmd_preprocess(_: argparse.Namespace) -> None:
    preprocessing_pipeline.run()


def _cmd_split(_: argparse.Namespace) -> None:
    preprocessing_pipeline.run_split()


def _cmd_understand(_: argparse.Namespace) -> None:
    eda_pipeline.run_understand()


def _cmd_eda(_: argparse.Namespace) -> None:
    eda_pipeline.run_eda_stage()


def _cmd_features(_: argparse.Namespace) -> None:
    feature_pipeline.run()


def _cmd_propose(_: argparse.Namespace) -> None:
    proposal_pipeline.run()


def _cmd_approve(args: argparse.Namespace) -> None:
    # Default (neither flag) approves both features and models.
    do_features = args.features or not (args.features or args.models)
    do_models = args.models or not (args.features or args.models)
    if do_features:
        approvals.approve_features()
    if do_models:
        approvals.approve_models()


def _cmd_train(_: argparse.Namespace) -> None:
    training_pipeline.run()


def _cmd_evaluate(_: argparse.Namespace) -> None:
    evaluation_pipeline.run()


def _cmd_explain(_: argparse.Namespace) -> None:
    reporting_pipeline.run_explain()


def _cmd_drivers(args: argparse.Namespace) -> None:
    # Dataset-agnostic sibling of `explain`: writes the driver table for any
    # project's champion without the housing-specific executive briefing.
    from src.explainability import drivers

    summary = drivers.write_driver_artifacts(force=getattr(args, "force", False))
    if summary is None:
        print(
            "No driver artifacts written. Either no champion is registered, the "
            "champion exposes no per-feature coefficients or importances, or a "
            "richer artifact from `explain` is already present (use --force to "
            "replace it)."
        )
        return
    print(
        f"Wrote {summary['n_features']} {summary['kind']} rows for "
        f"{summary['champion']}."
    )
    for path in summary["outputs"]:
        print(f"  {path}")


def _cmd_report(_: argparse.Namespace) -> None:
    reporting_pipeline.run_report()


def _cmd_churn_prep(_: argparse.Namespace) -> None:
    churn_pipeline.run_prep()


def _cmd_churn_all(_: argparse.Namespace) -> None:
    # One-shot churn pipeline: prep -> train -> evaluate. Requires the churn
    # workspace to be active (WORKBENCH_PROJECT=churn).
    churn_pipeline.run_prep()
    training_pipeline.run()
    evaluation_pipeline.run()


def _cmd_autorun(_: argparse.Namespace) -> None:
    # Bring-your-own-dataset one-shot. The active project (WORKBENCH_PROJECT)
    # must already be registered (workspaces/<slug>/dataset.json), e.g. via the
    # web app's upload flow or the GUI "New Project" page.
    #
    # Delegates to generic_pipeline.run_full so this command and the web upload
    # path cannot drift apart again - they previously kept separate call lists
    # and produced different deliverables for the same dataset.
    generic_pipeline.run_full()


def _cmd_trust(args: argparse.Namespace) -> None:
    trust_pipeline.run(split=args.split, groups=args.groups or None)


def _cmd_all(args: argparse.Namespace) -> None:
    # Delegates to the orchestrator with force=True so `all` keeps its prior
    # meaning exactly: run every stage end to end, regardless of what is
    # already on disk. Stage behavior itself is unchanged.
    stages = pipeline_builder.build_stages(dataset=args.dataset, download=args.download)
    orchestrator.run_pipeline(force=True, stages=stages)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    stages = pipeline_builder.build_stages(dataset=args.dataset, download=args.download)
    results = orchestrator.run_pipeline(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        force=args.force,
        stages=stages,
    )
    for result in results:
        logger.info(
            "Stage '%s': %s (%.2fs)",
            result["stage"],
            result["status"],
            result["duration_s"],
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI with one subcommand per pipeline stage."""

    parser = argparse.ArgumentParser(
        prog="workbench",
        description="AI Data Science Workbench - governed pipeline stages + orchestrator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    for name, handler, help_text in [
        ("ingest", _cmd_ingest, "Ingest, profile, and validate the dataset."),
        ("validate", _cmd_validate, "Run semantic-schema validation."),
        ("preprocess", _cmd_preprocess, "Clean, split, and fit train-only transforms."),
        ("split", _cmd_split, "Split the processed dataset into train/val/test."),
        ("understand", _cmd_understand, "Write the data-understanding report."),
        ("eda", _cmd_eda, "Run EDA and the analytical summary."),
        ("features", _cmd_features, "Write the feature proposal and execute it."),
        ("propose", _cmd_propose, "Write the governed model proposal."),
        ("approve", _cmd_approve, "Approve features and/or models (gates training)."),
        ("churn-prep", _cmd_churn_prep, "Prepare the churn project (ingest/clean/split/govern)."),
        ("churn-all", _cmd_churn_all, "Churn end to end: prep -> train -> evaluate."),
        ("autorun", _cmd_autorun, "Uploaded dataset end to end: auto prep -> train -> evaluate -> report."),
        ("train", _cmd_train, "Train every approved model."),
        ("evaluate", _cmd_evaluate, "Compare models and evaluate the champion on test."),
        ("explain", _cmd_explain, "Champion coefficients + executive briefing."),
        ("drivers", _cmd_drivers, "Champion driver table only (works for any dataset)."),
        ("trust", _cmd_trust, "Subgroup fairness, permutation importance, calibration, model card."),
        ("report", _cmd_report, "Write the self-contained HTML project report."),
        ("all", _cmd_all, "Run the full pipeline chain end to end (orchestrated, always reruns)."),
        ("pipeline", _cmd_pipeline, "Run an orchestrated stage range (--from/--to/--force)."),
    ]:
        sub = _add(name, handler, help_text)
        if name == "drivers":
            sub.add_argument(
                "--force", action="store_true",
                help="Replace an existing artifact written by `explain`.",
            )
        if name == "trust":
            sub.add_argument(
                "--split", default="test", choices=["test", "validation"],
                help="Split to audit (default: test, the only unbiased estimate; "
                "the champion is already frozen, so this reports rather than selects).",
            )
            sub.add_argument(
                "--groups", nargs="*", default=None, metavar="COLUMN",
                help="Subgroup columns to audit (default: auto-detect every column "
                "with 2-10 distinct non-float values).",
            )
        if name == "approve":
            sub.add_argument(
                "--features", action="store_true",
                help="Approve features only (default: approve both).",
            )
            sub.add_argument(
                "--models", action="store_true",
                help="Approve models only (default: approve both).",
            )
        if name == "pipeline":
            sub.add_argument(
                "--from", dest="from_stage", default=None,
                choices=pipeline_builder.STAGE_NAMES,
                help="First stage to run (default: the first registered stage, 'ingest').",
            )
            sub.add_argument(
                "--to", dest="to_stage", default=None,
                choices=pipeline_builder.STAGE_NAMES,
                help="Last stage to run (default: the last registered stage, 'report').",
            )
            sub.add_argument(
                "--force", action="store_true",
                help="Rerun every stage in range even if its outputs already exist "
                "(default: skip a stage whose declared outputs are already present).",
            )
        if name in {"ingest", "validate", "all", "pipeline"}:
            sub.add_argument(
                "--dataset", default="california_housing",
                help="Registered dataset name (default: california_housing).",
            )
        if name in {"ingest", "all", "pipeline"}:
            sub.add_argument(
                "--download", action="store_true",
                help="Force a fresh Kaggle download instead of using the cache.",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as exc:  # noqa: BLE001 - surface a clean message + exit code.
        logger.error("Stage '%s' failed: %s", args.command, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
