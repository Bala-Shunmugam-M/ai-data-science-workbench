"""
src.workflow.pipeline_builder
==============================

PURPOSE
-------
Declarative stage registry for the Milestone-3 orchestrator. Each
:class:`StageSpec` wraps the SAME ``run()`` function ``main.py`` already
calls for that stage, plus the input files it requires and the output files
it produces. The orchestrator never reimplements stage logic - it only reads
this registry to sequence stages, check dependencies, and report status.

PIPELINE POSITION
-----------------
Read by :mod:`src.workflow.orchestrator` and by ``main.py`` (the ``pipeline``
and ``all`` subcommands). The registry order IS the canonical pipeline order:

    ingest -> preprocess -> understand -> eda -> features -> propose ->
    approve -> train -> evaluate -> explain -> trust -> report

(``validate`` and ``split`` remain standalone ``main.py`` sub-steps of
``ingest``/``preprocess`` respectively and are not separate orchestrator
stages, since their outputs are a subset of those two stages'.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config.paths import (
    ENGINEERED_TEST_PATH,
    ENGINEERED_TRAIN_PATH,
    ENGINEERED_VALIDATION_PATH,
    EXECUTIVE_BRIEFING_PATH,
    EXPLAINABILITY_DIR,
    FAIRNESS_JSON_PATH,
    FEATURE_APPROVAL_PATH,
    FEATURE_PROPOSAL_PATH,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_CARD_MD_PATH,
    MODEL_APPROVAL_PATH,
    MODEL_COMPARISON_CSV,
    MODEL_EVALUATION_REPORT_PATH,
    MODEL_PROPOSAL_PATH,
    MODEL_REGISTRY_PATH,
    PROCESSED_HOUSING_PATH,
    PROJECT_REPORT_PATH,
    RAW_HOUSING_PATH,
    RESULTS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from src.governance import approvals
from src.pipelines import (
    eda_pipeline,
    evaluation_pipeline,
    feature_pipeline,
    ingestion_pipeline,
    preprocessing_pipeline,
    proposal_pipeline,
    reporting_pipeline,
    training_pipeline,
    trust_pipeline,
)


@dataclass(frozen=True)
class StageSpec:
    """
    One orchestrator-managed stage.

    ``inputs`` are files that must already exist before ``run`` is called
    (checked against the ``requires`` prerequisite stage names for the
    refusal message). ``outputs`` are files ``run`` is expected to produce;
    the orchestrator uses their presence to support resume/skip.
    """

    name: str
    description: str
    run: Callable[[], Any]
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


def _approve_both() -> dict[str, Any]:
    """Approve features then models (the ``approve`` stage does both by default)."""

    return {
        "features": approvals.approve_features(),
        "models": approvals.approve_models(),
    }


def build_stages(
    dataset: str = "california_housing",
    download: bool = False,
) -> list[StageSpec]:
    """
    Build the ordered stage registry.

    ``dataset``/``download`` are threaded into the ``ingest`` stage only,
    mirroring ``main.py ingest --dataset ... [--download]``.
    """

    return [
        StageSpec(
            name="ingest",
            description="Ingest, profile, and validate the dataset.",
            run=lambda: ingestion_pipeline.run(dataset, prefer_cache=not download),
            inputs=[],
            outputs=[RAW_HOUSING_PATH],
            requires=[],
        ),
        StageSpec(
            name="preprocess",
            description="Clean, split, and fit train-only transforms.",
            run=preprocessing_pipeline.run,
            inputs=[RAW_HOUSING_PATH],
            outputs=[PROCESSED_HOUSING_PATH, TRAIN_PATH, VALIDATION_PATH, TEST_PATH],
            requires=["ingest"],
        ),
        StageSpec(
            name="understand",
            description="Write the data-understanding report.",
            run=eda_pipeline.run_understand,
            inputs=[TRAIN_PATH],
            outputs=[RESULTS_DIR / "data_understanding" / "dataset_summary.csv"],
            requires=["preprocess"],
        ),
        StageSpec(
            name="eda",
            description="Run EDA and the analytical summary.",
            run=eda_pipeline.run_eda_stage,
            inputs=[TRAIN_PATH],
            outputs=[RESULTS_DIR / "eda" / "eda_report.txt"],
            requires=["preprocess"],
        ),
        StageSpec(
            name="features",
            description="Write the feature proposal and execute it.",
            run=feature_pipeline.run,
            inputs=[TRAIN_PATH, VALIDATION_PATH, TEST_PATH],
            outputs=[
                FEATURE_PROPOSAL_PATH,
                ENGINEERED_TRAIN_PATH,
                ENGINEERED_VALIDATION_PATH,
                ENGINEERED_TEST_PATH,
            ],
            requires=["preprocess"],
        ),
        StageSpec(
            name="propose",
            description="Write the governed model proposal.",
            run=proposal_pipeline.run,
            inputs=[ENGINEERED_TRAIN_PATH],
            outputs=[MODEL_PROPOSAL_PATH],
            requires=["features"],
        ),
        StageSpec(
            name="approve",
            description="Approve features and models (gates training).",
            run=_approve_both,
            inputs=[FEATURE_PROPOSAL_PATH, MODEL_PROPOSAL_PATH],
            outputs=[FEATURE_APPROVAL_PATH, MODEL_APPROVAL_PATH],
            requires=["features", "propose"],
        ),
        StageSpec(
            name="train",
            description="Train every approved model.",
            run=training_pipeline.run,
            inputs=[
                MODEL_APPROVAL_PATH,
                FEATURE_APPROVAL_PATH,
                ENGINEERED_TRAIN_PATH,
                ENGINEERED_VALIDATION_PATH,
            ],
            outputs=[MODEL_REGISTRY_PATH],
            requires=["approve"],
        ),
        StageSpec(
            name="evaluate",
            description="Compare models and evaluate the champion on test.",
            run=evaluation_pipeline.run,
            inputs=[
                MODEL_REGISTRY_PATH,
                ENGINEERED_VALIDATION_PATH,
                ENGINEERED_TEST_PATH,
            ],
            outputs=[
                MODEL_COMPARISON_CSV,
                MODEL_EVALUATION_REPORT_PATH,
                FINAL_MODEL_SELECTION_PATH,
            ],
            requires=["train"],
        ),
        StageSpec(
            name="explain",
            description="Champion coefficients + executive briefing.",
            run=reporting_pipeline.run_explain,
            inputs=[FINAL_MODEL_SELECTION_PATH],
            outputs=[
                EXECUTIVE_BRIEFING_PATH,
                EXPLAINABILITY_DIR / "coefficient_interpretation.csv",
            ],
            requires=["evaluate"],
        ),
        StageSpec(
            name="trust",
            description="Subgroup fairness, permutation importance, calibration, model card.",
            run=trust_pipeline.run,
            inputs=[FINAL_MODEL_SELECTION_PATH, ENGINEERED_TEST_PATH],
            outputs=[FAIRNESS_JSON_PATH, MODEL_CARD_MD_PATH],
            requires=["evaluate"],
        ),
        StageSpec(
            name="report",
            description="Write the self-contained HTML project report.",
            run=reporting_pipeline.run_report,
            inputs=[EXECUTIVE_BRIEFING_PATH],
            outputs=[PROJECT_REPORT_PATH],
            # Deliberately ``explain``, not ``trust``: the report predates the trust
            # stage and must not start failing when a trust artifact is absent.
            # Trust runs before it in registry order, so a full run still picks it up.
            requires=["explain"],
        ),
    ]


# Canonical stage-name order, exposed for ``main.py``'s ``--from``/``--to`` choices.
STAGE_NAMES: list[str] = [stage.name for stage in build_stages()]
