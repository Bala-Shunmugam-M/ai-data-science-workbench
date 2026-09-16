"""
src.pipelines.trust_pipeline
============================

PURPOSE
-------
Stage 11: Trust. Runs the four audits that turn a scored model into a model
someone can be accountable for - subgroup fairness, permutation importance,
probability calibration - and assembles them into a model card.

PIPELINE POSITION
-----------------
    evaluation -> explainability -> [trust] -> reporting

It runs after ``explain`` rather than before it so the model card can quote the
driver table, and after ``evaluate`` because it needs a champion. With no
champion registered it logs and returns instead of raising: that is the pipeline
being incomplete, not broken.

GOVERNANCE
----------
The champion is already frozen when this stage runs, so reading the test split
again reports on a decision rather than influencing one - see :mod:`src.trust`.
The run is audit-logged and recorded as a lineage node, like every other stage.

OUTPUTS
-------
    artifacts/trust/subgroup_fairness.{json,csv}
    artifacts/trust/permutation_importance.{json,csv}
    artifacts/trust/calibration.json
    artifacts/trust/model_card.{md,json}
    artifacts/trust/figures/*.png
"""

from __future__ import annotations

from typing import Any

from config.paths import (
    CALIBRATION_JSON_PATH,
    FAIRNESS_JSON_PATH,
    MODEL_CARD_JSON_PATH,
    MODEL_CARD_MD_PATH,
    ROBUST_IMPORTANCE_JSON_PATH,
    ensure_dirs,
)
from src.governance import audit_logger, lineage_tracker
from src.trust import calibration, fairness, model_card, robust_importance
from src.trust.subject import SPLIT_PATHS, load_subject
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(
    *,
    split: str = "test",
    groups: list[str] | None = None,
    actor: str | None = None,
) -> dict[str, Any] | None:
    """
    Run the trust stage and return a summary, or ``None`` with no champion.

    ``groups`` overrides the auto-detected subgroup columns and is threaded
    straight through to :func:`src.trust.fairness.run`.
    """

    ensure_dirs()
    logger.info("=== Trust audit (fairness, importance, calibration, model card) ===")

    subject = load_subject(split=split)
    if subject is None:
        logger.info("No champion registered. Run 'python main.py evaluate' first.")
        return None

    fairness_payload = fairness.run(subject, groups=groups)
    importance_payload = robust_importance.run(subject)
    calibration_payload = calibration.run(subject)
    card = model_card.run()

    summary: dict[str, Any] = {
        "champion": subject.label,
        "split": subject.split,
        "columns_audited": fairness_payload["columns_audited"],
        "n_features_ranked": importance_payload["n_features"],
        "n_features_above_noise": importance_payload["n_distinguishable_from_noise"],
        "calibration_assessed": bool(calibration_payload.get("assessed")),
        "model_card": str(MODEL_CARD_MD_PATH) if card else None,
    }

    audit_logger.log_event(
        "trust_audit_run",
        {
            "champion": subject.label,
            "split": subject.split,
            "columns_audited": fairness_payload["columns_audited"],
            "calibration_assessed": summary["calibration_assessed"],
            # Recorded so a later reader can see whether the audit found evidence
            # of dependence at all, not merely that it ran.
            "n_features_above_noise": summary["n_features_above_noise"],
        },
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    lineage_tracker.record_node(
        "trust_audit",
        input_paths=[SPLIT_PATHS[subject.split]],
        output_paths=[
            FAIRNESS_JSON_PATH,
            ROBUST_IMPORTANCE_JSON_PATH,
            CALIBRATION_JSON_PATH,
            MODEL_CARD_MD_PATH,
            MODEL_CARD_JSON_PATH,
        ],
        script="src/pipelines/trust_pipeline.py",
        params={"split": subject.split, "champion": subject.label},
        extra=summary,
    )

    logger.info(
        "Trust audit complete. Subgroups audited: %s. Calibration: %s.",
        ", ".join(summary["columns_audited"]) or "none found",
        "assessed" if summary["calibration_assessed"] else "not applicable",
    )
    return summary
