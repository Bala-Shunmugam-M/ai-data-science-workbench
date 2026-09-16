"""
src.explainability.shap_explainer
=================================

PURPOSE
-------
Optional, dependency-guarded SHAP explanations for the champion model. SHAP is
**not** a project requirement (it is intentionally absent from
``requirements.txt``): for a linear-family model the standardized coefficients in
:mod:`src.explainability.interpretation` already give an exact, additive,
per-feature attribution, so SHAP adds cost without adding insight for this scope.

This module exists so a team that *does* install ``shap`` can generate a SHAP
summary without changing any other code. Every entry point degrades gracefully:
if ``shap`` cannot be imported, the functions return a clear "skipped" record
rather than raising, and the pipeline continues.

PIPELINE POSITION
-----------------
    evaluation -> [explainability (optional SHAP)] -> reporting

OUTPUTS (only when ``shap`` is installed)
-----------------------------------------
    artifacts/explainability/shap_summary.png
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from config.paths import ENGINEERED_VALIDATION_PATH, EXPLAINABILITY_DIR, ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SHAP_SUMMARY_PNG: Path = EXPLAINABILITY_DIR / "shap_summary.png"


def shap_available() -> bool:
    """Return True only if the optional ``shap`` package is importable."""

    return importlib.util.find_spec("shap") is not None


def explain_champion_with_shap(
    *,
    max_samples: int = 500,
    validation_path: Path = ENGINEERED_VALIDATION_PATH,
) -> dict[str, Any]:
    """
    Produce a SHAP summary plot for the champion, if ``shap`` is installed.

    This is a best-effort optional extra. When ``shap`` is not available (the
    default), it logs and returns ``{"status": "skipped", ...}`` without raising,
    so callers can invoke it unconditionally.
    """

    if not shap_available():
        logger.info(
            "shap not installed; skipping optional SHAP explanation "
            "(standardized coefficients already provide exact linear attributions)."
        )
        return {
            "status": "skipped",
            "reason": "shap-not-installed",
            "hint": "pip install shap to enable this optional explanation.",
        }

    # Imports are deferred so the module loads with or without shap present.
    import joblib  # noqa: WPS433
    import shap  # noqa: WPS433  (optional dependency)

    from src.artifacts import model_registry  # noqa: WPS433
    from src.utils.file_utils import load_csv  # noqa: WPS433

    champion = model_registry.get_champion()
    if champion is None:
        return {"status": "skipped", "reason": "no-champion"}

    bundle = joblib.load(champion["model_path"])
    predictor_columns = bundle["predictor_columns"]
    validation_df = load_csv(validation_path)
    design = bundle["preprocessor"].transform(validation_df[predictor_columns])
    sample = design.sample(min(max_samples, len(design)), random_state=42)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.LinearExplainer(bundle["estimator"], sample)
        shap_values = explainer.shap_values(sample)

        ensure_dir(EXPLAINABILITY_DIR)
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PNG, dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001 - optional path must never break the run.
        logger.warning("Optional SHAP explanation failed: %s", exc)
        return {"status": "error", "reason": str(exc)}

    logger.info("SHAP summary written -> %s", SHAP_SUMMARY_PNG)
    return {"status": "ok", "shap_summary_png": str(SHAP_SUMMARY_PNG)}
