"""
src.model_evaluation.evaluator
==============================

PURPOSE
-------
Evaluate all registered trained models on the validation split, promote the
champion (best validation *selection metric* — RMSE for regression, ROC-AUC for
classification), and evaluate ONLY the champion once on the untouched test
split. Produces the comparison table, the evaluation report, champion
diagnostics (residuals for regression; confusion matrix + ROC for
classification), and the final model-selection artifact.

GOVERNANCE
----------
The test split is loaded exactly once and only the champion's predictions are
scored against it, preserving the professor's untouched-test doctrine. The
selection is audit-logged and a lineage node is recorded.

PIPELINE POSITION
-----------------
    training -> [evaluation] -> explainability -> reporting
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are written, never displayed.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from config import active  # noqa: E402
from config.constants import TARGET_COLUMN  # noqa: E402
from config.paths import (  # noqa: E402
    ENGINEERED_TEST_PATH,
    ENGINEERED_VALIDATION_PATH,
    EVALUATION_FIGURES_DIR,
    FINAL_MODEL_SELECTION_PATH,
    MODEL_COMPARISON_CSV,
    MODEL_EVALUATION_REPORT_PATH,
    ensure_dir,
)
from src.artifacts import model_registry  # noqa: E402
from src.governance import GovernanceError, audit_logger, lineage_tracker  # noqa: E402
from src.model_evaluation import comparator  # noqa: E402
from src.model_training import metrics as metric_helpers  # noqa: E402
from src.model_training.design_matrix import class_scores, encode_target  # noqa: E402
from src.utils.common import utc_timestamp  # noqa: E402
from src.utils.file_utils import load_csv, save_csv, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def select_champion(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Pick the best row by ``metric``, direction-aware and NaN-safe.

    Module level, not a closure inside :func:`run`, so the choice can be tested
    without training a model or writing an artifact - see
    ``tests/test_metric_selection.py``. The original version of this logic
    lived inline and shipped a bug that no test could reach.

    NaN is treated exactly like a missing value. ``metric_helpers.roc_auc``
    returns NaN by design when the metric is undefined (a class absent from the
    validation split), and every NaN comparison is ``False`` - so ``max()`` /
    ``min()`` never replace a NaN once it is the running best. A single
    undefined score would otherwise win champion selection outright, promoting
    an unevaluated model over a demonstrably better one, silently.
    """

    def key(row: dict[str, Any]) -> float:
        value = row["validation_metrics"].get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return metric_helpers.worst_value(metric)
        return value

    higher = metric_helpers.METRIC_HIGHER_IS_BETTER.get(metric, False)
    return (max if higher else min)(rows, key=key)


def _predict(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Transform ``df`` with the bundled preprocessor and predict (labels/values)."""

    predictor_columns = bundle["predictor_columns"]
    matrix = bundle["preprocessor"].transform(df[predictor_columns])
    return bundle["estimator"].predict(matrix)


def _score(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray | None:
    """
    Probability scores for a classifier bundle, else None. Binary yields
    positive-class probabilities; multiclass yields the full proba matrix.
    """

    matrix = bundle["preprocessor"].transform(df[bundle["predictor_columns"]])
    return class_scores(bundle["estimator"], matrix)


def _metrics(bundle: dict[str, Any], df: pd.DataFrame, y_true: pd.Series, task: str) -> dict[str, float]:
    if task == "classification":
        return metric_helpers.all_metrics(
            y_true, _predict(bundle, df), _score(bundle, df), task="classification"
        )
    return metric_helpers.all_metrics(y_true, _predict(bundle, df))


def _residual_figures(
    y_true: pd.Series, y_pred: np.ndarray, champion_label: str, figures_dir: Path
) -> dict[str, str]:
    """Write the three champion residual diagnostics; return their paths."""

    ensure_dir(figures_dir)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=6, alpha=0.3, edgecolor="none")
    ax.axhline(0.0, color="crimson", linewidth=1.2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals vs Predicted - {champion_label}")
    fig.tight_layout()
    rp = figures_dir / "residuals_vs_predicted.png"
    fig.savefig(rp, dpi=120)
    plt.close(fig)
    paths["residuals_vs_predicted"] = str(rp)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0.0, color="crimson", linewidth=1.2)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution - {champion_label}")
    fig.tight_layout()
    rh = figures_dir / "residual_histogram.png"
    fig.savefig(rh, dpi=120)
    plt.close(fig)
    paths["residual_histogram"] = str(rh)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, edgecolor="none")
    lims = [
        float(min(np.min(y_true), np.min(y_pred))),
        float(max(np.max(y_true), np.max(y_pred))),
    ]
    ax.plot(lims, lims, color="crimson", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual - {champion_label}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    pa = figures_dir / "predicted_vs_actual.png"
    fig.savefig(pa, dpi=120)
    plt.close(fig)
    paths["predicted_vs_actual"] = str(pa)
    return paths


def _classification_figures(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    champion_label: str,
    figures_dir: Path,
) -> dict[str, str]:
    """Write confusion matrix + ROC curve for the champion classifier."""

    ensure_dir(figures_dir)
    paths: dict[str, str] = {}

    # Class labels present across truth/prediction, in a stable order.
    labels = sorted(set(np.unique(np.asarray(y_true))) | set(np.unique(np.asarray(y_pred))))
    binary = len(labels) <= 2
    if binary and set(labels) <= {0, 1}:
        tick_labels = ["No", "Yes"]
        labels = [0, 1]
    else:
        tick_labels = [str(v) for v in labels]

    # Confusion matrix (N x N).
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    size = max(5.0, 1.1 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), labels=[f"Pred: {t}" for t in tick_labels], rotation=45, ha="right")
    ax.set_yticks(range(n), labels=[f"Actual: {t}" for t in tick_labels])
    # Annotate cells only while they stay readable.
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13 if n <= 4 else 9,
                )
    ax.set_title(f"Confusion Matrix - {champion_label}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cmp_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cmp_path, dpi=120)
    plt.close(fig)
    paths["confusion_matrix"] = str(cmp_path)

    # ROC curve: single curve for binary, one-vs-rest per class for multiclass.
    score = None if y_score is None else np.asarray(y_score)
    if score is not None and len(np.unique(np.asarray(y_true))) > 1:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        drew = False
        if score.ndim > 1 and score.shape[1] > 2:
            # One-vs-rest: binarise the truth for each class column in turn.
            classes = sorted(np.unique(np.asarray(y_true)))
            for idx, cls in enumerate(classes):
                if idx >= score.shape[1]:
                    break
                truth = (np.asarray(y_true) == cls).astype(int)
                if len(np.unique(truth)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(truth, score[:, idx])
                ax.plot(fpr, tpr, linewidth=1.8, label=f"{cls} (AUC={metric_helpers.roc_auc(truth, score[:, idx]):.3f})")
                drew = True
            ax.set_title(f"ROC Curves (one-vs-rest) - {champion_label}")
        else:
            positive = score[:, 1] if score.ndim > 1 else score
            fpr, tpr, _ = roc_curve(y_true, positive)
            auc = metric_helpers.roc_auc(y_true, positive)
            ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC={auc:.3f})")
            ax.set_title(f"ROC Curve - {champion_label}")
            drew = True
        if drew:
            ax.plot([0, 1], [0, 1], color="crimson", linewidth=1, linestyle="--", label="Chance")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            roc_path = figures_dir / "roc_curve.png"
            fig.savefig(roc_path, dpi=120)
            paths["roc_curve"] = str(roc_path)
        plt.close(fig)
    return paths


def evaluate(
    *,
    validation_path: Path = ENGINEERED_VALIDATION_PATH,
    test_path: Path = ENGINEERED_TEST_PATH,
    actor: str | None = None,
) -> dict[str, Any]:
    """Run the full evaluation stage and return the evaluation report dict."""

    registered = model_registry.list_models()
    if not registered:
        raise GovernanceError(
            "No trained models in the registry. Run 'python main.py train' first."
        )

    task = active.task()
    metric = active.selection_metric()
    positive_class = active.positive_class()
    target = active.target_column()

    validation_df = load_csv(validation_path)
    y_val = encode_target(validation_df[target], positive_class)

    # --- Validation comparison over ALL registered models. ---
    comparison_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registered:
        bundle = joblib.load(entry["model_path"])
        key = (entry["name"], entry["version"])
        bundles[key] = bundle
        val_metrics = _metrics(bundle, validation_df, y_val, task)
        comparison_rows.append(
            {"name": entry["name"], "version": entry["version"], "validation_metrics": val_metrics}
        )

    table = comparator.comparison_table(comparison_rows)
    save_csv(table, MODEL_COMPARISON_CSV)

    # --- Champion = best validation selection metric (direction-aware). ---
    champion_row = select_champion(comparison_rows, metric)
    champion_name = champion_row["name"]
    champion_version = champion_row["version"]
    champion_label = f"{champion_name} {champion_version}"
    champion_bundle = bundles[(champion_name, champion_version)]

    # --- The ONLY use of the untouched test split: score the champion once. ---
    test_df = load_csv(test_path)
    y_test = encode_target(test_df[target], positive_class)
    champion_test_pred = _predict(champion_bundle, test_df)
    champion_test_metrics = _metrics(champion_bundle, test_df, y_test, task)

    if task == "classification":
        figures = _classification_figures(
            y_test, champion_test_pred, _score(champion_bundle, test_df),
            champion_label, EVALUATION_FIGURES_DIR,
        )
    else:
        figures = _residual_figures(
            y_test, champion_test_pred, champion_label, EVALUATION_FIGURES_DIR
        )

    model_registry.set_champion(champion_name, champion_version)
    narrative = comparator.selection_narrative(table, champion_row, champion_test_metrics)

    report: dict[str, Any] = {
        "artifact": "model_evaluation_report",
        "generated_at": utc_timestamp(),
        "task": task,
        "selection_metric": metric,
        "n_models_compared": len(comparison_rows),
        "validation_comparison": comparison_rows,
        "champion": {
            "name": champion_name,
            "version": champion_version,
            "validation_metrics": champion_row["validation_metrics"],
            "test_metrics": champion_test_metrics,
        },
        "champion_figures": figures,
        "selection_narrative": narrative,
    }
    # Back-compat: the regression showcase's report generator + GUI read
    # ``residual_figures``; keep emitting it so housing is byte-for-byte unchanged.
    if task == "regression":
        report["residual_figures"] = figures
    save_json(report, MODEL_EVALUATION_REPORT_PATH)

    final_selection: dict[str, Any] = {
        "artifact": "final_model_selection",
        "generated_at": utc_timestamp(),
        "task": task,
        "selection_metric": metric,
        "champion_name": champion_name,
        "champion_version": champion_version,
        "test_metrics": champion_test_metrics,
        "validation_metrics": champion_row["validation_metrics"],
        "selection_rationale": narrative,
        "model_path": model_registry.get_model(champion_name, champion_version)["model_path"],
    }
    # Preserve the regression showcase's flat keys for its existing report/GUI.
    if task == "regression":
        final_selection.update(
            {
                "test_rmse": champion_test_metrics.get("rmse"),
                "test_mae": champion_test_metrics.get("mae"),
                "test_r2": champion_test_metrics.get("r2"),
                "test_mape": champion_test_metrics.get("mape"),
                "validation_rmse": champion_row["validation_metrics"].get("rmse"),
            }
        )
    save_json(final_selection, FINAL_MODEL_SELECTION_PATH)

    audit_logger.log_event(
        "model_evaluation_run",
        {
            "n_models_compared": len(comparison_rows),
            "champion": champion_label,
            f"champion_test_{metric}": champion_test_metrics.get(metric),
        },
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    audit_logger.log_event(
        "final_model_selected",
        {"champion": champion_label, "test_metrics": champion_test_metrics},
        actor=actor or audit_logger.DEFAULT_ACTOR,
    )
    champion_model_path = model_registry.get_model(champion_name, champion_version)["model_path"]
    lineage_tracker.record_node(
        "model_evaluation",
        input_paths=[Path(champion_model_path), validation_path, test_path],
        output_paths=[MODEL_COMPARISON_CSV, MODEL_EVALUATION_REPORT_PATH, FINAL_MODEL_SELECTION_PATH],
        script="src/model_evaluation/evaluator.py",
        params={"champion": champion_label, "selection_metric": metric},
        extra={"champion": champion_label, "test_metrics": champion_test_metrics},
    )

    summary = " ".join(f"{k}={v:.4f}" for k, v in champion_test_metrics.items())
    logger.info("Evaluation complete. Champion=%s | test %s", champion_label, summary)
    return report
