"""
tests.test_model_pipeline
=========================

Milestone-2 tests for the governed modelling chain. Every test uses synthetic
data and a fully redirected temporary workspace, so nothing here touches the
network or the project's real artifacts under ``models/``, ``artifacts/`` or
``governance/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.constants import ALPHA_GRID_SIZE, TARGET_COLUMN
from src.artifacts import experiment_tracker, model_registry
from src.governance import (
    GovernanceError,
    approvals,
    audit_logger,
    lineage_tracker,
)
from src.model_evaluation import evaluator
from src.model_training import metrics as metric_helpers
from src.model_training import trainer
from src.model_training.design_matrix import ModelingPreprocessor, split_x_y
from src.utils.file_utils import save_json


# ---------------------------------------------------------------------------
# Synthetic engineered data + a redirected workspace.
# ---------------------------------------------------------------------------
def _engineered_frame(n: int, seed: int) -> pd.DataFrame:
    """Small schema-conformant frame with a genuinely predictable target."""

    rng = np.random.default_rng(seed)
    median_income = rng.uniform(0.5, 12.0, n)
    households = rng.integers(50, 600, n).astype(float)
    population = households * rng.uniform(1.5, 4.0, n)
    total_rooms = households * rng.uniform(4.0, 8.0, n)
    total_bedrooms = households * rng.uniform(0.8, 1.5, n)
    ocean = rng.choice(["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"], n)
    # Target is a clean linear function of income + noise so models can learn it.
    target = 40_000 + 42_000 * median_income + rng.normal(0, 15_000, n)
    return pd.DataFrame(
        {
            "longitude": rng.uniform(-124.0, -114.0, n),
            "latitude": rng.uniform(32.0, 42.0, n),
            "housing_median_age": rng.integers(1, 52, n).astype(float),
            "total_rooms": np.round(total_rooms),
            "total_bedrooms": np.round(total_bedrooms),
            "population": np.round(population),
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean,
            TARGET_COLUMN: target,
        }
    )


_PREDICTORS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity",
]


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """
    Redirect every file the training/evaluation chain writes into ``tmp_path``.

    Functions that take a keyword-only ``path=`` default are redirected by
    patching their ``__kwdefaults__`` (reversibly, via monkeypatch); modules that
    reference path constants directly are redirected via ``setattr``.
    """

    # Engineered splits on disk.
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    _engineered_frame(600, 1).to_csv(train_path, index=False)
    _engineered_frame(200, 2).to_csv(val_path, index=False)
    _engineered_frame(200, 3).to_csv(test_path, index=False)

    # Redirect target files.
    registry_path = tmp_path / "model_registry.json"
    experiments_path = tmp_path / "experiments.jsonl"
    audit_path = tmp_path / "audit_log.jsonl"
    lineage_path = tmp_path / "lineage.json"
    feature_approval = tmp_path / "feature_approval.json"
    model_approval = tmp_path / "model_approval.json"

    def redirect(func, key, value):
        # Keyword-only defaults live in __kwdefaults__; a plain positional default
        # for a single-argument reader lives in __defaults__.
        if func.__kwdefaults__ and key in func.__kwdefaults__:
            monkeypatch.setitem(func.__kwdefaults__, key, value)
        elif func.__defaults__ and len(func.__defaults__) == 1:
            monkeypatch.setattr(func, "__defaults__", (value,))
        else:
            raise AssertionError(f"{func.__name__} has no redirectable '{key}' default")

    # model_registry.*(path=...)
    for fn in (
        model_registry.register_model, model_registry.list_models,
        model_registry.get_model, model_registry.next_version,
        model_registry.set_champion, model_registry.get_champion,
    ):
        redirect(fn, "path", registry_path)
    redirect(experiment_tracker.log_experiment, "path", experiments_path)
    redirect(audit_logger.log_event, "path", audit_path)
    redirect(lineage_tracker.record_node, "path", lineage_path)

    # approvals readers.
    for fn in (
        approvals.approved_models, approvals.require_model_approved,
        approvals.load_model_approval,
    ):
        redirect(fn, "path", model_approval)
    for fn in (
        approvals.approved_predictor_columns, approvals.load_feature_approval,
    ):
        redirect(fn, "path", feature_approval)

    # Modules that use path constants directly.
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(evaluator, "MODEL_COMPARISON_CSV", tmp_path / "model_comparison.csv")
    monkeypatch.setattr(evaluator, "MODEL_EVALUATION_REPORT_PATH", tmp_path / "eval_report.json")
    monkeypatch.setattr(evaluator, "FINAL_MODEL_SELECTION_PATH", tmp_path / "final_selection.json")
    monkeypatch.setattr(evaluator, "EVALUATION_FIGURES_DIR", tmp_path / "figures")

    return {
        "tmp": tmp_path,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "feature_approval": feature_approval,
        "model_approval": model_approval,
        "registry_path": registry_path,
    }


def _approve(ws, models):
    """Write minimal feature+model approval files into the workspace."""

    save_json(
        {
            "artifact": "feature_approval",
            "approved_predictor_columns": _PREDICTORS,
            "target_column": TARGET_COLUMN,
        },
        ws["feature_approval"],
    )
    save_json(
        {"artifact": "model_approval", "approved_models": models},
        ws["model_approval"],
    )


# ---------------------------------------------------------------------------
# 1. Governance refuses to train an unapproved model.
# ---------------------------------------------------------------------------
def test_governance_refuses_unapproved_model(workspace):
    _approve(workspace, ["Ridge"])  # LinearRegression NOT approved.

    approvals.require_model_approved("Ridge")  # approved -> no raise.
    with pytest.raises(GovernanceError):
        approvals.require_model_approved("LinearRegression")


def test_training_raises_when_nothing_approved(workspace):
    _approve(workspace, [])  # empty approval list.
    with pytest.raises(GovernanceError):
        trainer.train_all(
            train_path=workspace["train_path"],
            validation_path=workspace["val_path"],
        )


# ---------------------------------------------------------------------------
# 2. The trainer excludes the target from the predictors.
# ---------------------------------------------------------------------------
def test_split_x_y_excludes_target():
    df = _engineered_frame(50, 7)

    x, y = split_x_y(df, _PREDICTORS)
    assert TARGET_COLUMN not in x.columns
    assert y.name == TARGET_COLUMN
    assert list(x.columns) == _PREDICTORS

    # Passing the target as a predictor is rejected outright.
    with pytest.raises(ValueError):
        split_x_y(df, _PREDICTORS + [TARGET_COLUMN])


def test_approved_predictor_columns_never_include_target(workspace):
    _approve(workspace, ["Ridge"])
    columns = approvals.approved_predictor_columns()
    assert TARGET_COLUMN not in columns


# ---------------------------------------------------------------------------
# 3. Hyperparameter tuning selects alpha on the validation set.
# ---------------------------------------------------------------------------
def test_tuning_selects_best_validation_alpha():
    train = _engineered_frame(400, 11)
    val = _engineered_frame(150, 12)
    pre = ModelingPreprocessor(predictor_columns=_PREDICTORS).fit(train[_PREDICTORS])
    x_tr, y_tr = pre.transform(train[_PREDICTORS]), train[TARGET_COLUMN]
    x_va, y_va = pre.transform(val[_PREDICTORS]), val[TARGET_COLUMN]

    best_params, history = trainer._tune(
        "Ridge", x_tr, y_tr, x_va, y_va,
        task="regression", selection_metric="rmse",
    )

    # The full log-spaced grid was swept on the validation split.
    assert len(history) == ALPHA_GRID_SIZE
    # The chosen alpha is exactly the one with the lowest VALIDATION rmse.
    best_by_val = min(history, key=lambda h: h["validation_rmse"])
    assert best_params["alpha"] == pytest.approx(best_by_val["alpha"])


# ---------------------------------------------------------------------------
# 4. Model registry round-trip.
# ---------------------------------------------------------------------------
def test_model_registry_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    assert model_registry.next_version("Ridge", path=path) == "v001"

    entry = model_registry.register_model(
        {
            "name": "Ridge",
            "version": "v001",
            "model_path": "models/Ridge/v001/model.joblib",
            "validation_metrics": {"rmse": 100.0},
        },
        path=path,
    )
    assert entry["name"] == "Ridge"
    assert model_registry.next_version("Ridge", path=path) == "v002"

    fetched = model_registry.get_model("Ridge", "v001", path=path)
    assert fetched["model_path"] == "models/Ridge/v001/model.joblib"

    model_registry.set_champion("Ridge", "v001", path=path)
    champ = model_registry.get_champion(path=path)
    assert champ["name"] == "Ridge" and champ["is_champion"] is True

    with pytest.raises(KeyError):
        model_registry.set_champion("Ghost", "v001", path=path)


# ---------------------------------------------------------------------------
# 5. Metrics correctness on a hand-computed example.
# ---------------------------------------------------------------------------
def test_metrics_hand_computed():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    # By hand: MAE = (0.5+0.5+0+1)/4 = 0.5; MSE = (0.25+0.25+0+1)/4 = 0.375.
    assert metric_helpers.mae(y_true, y_pred) == pytest.approx(0.5)
    assert metric_helpers.rmse(y_true, y_pred) == pytest.approx(0.375 ** 0.5)
    # R2 for this canonical example is 0.948608...
    assert metric_helpers.r2(y_true, y_pred) == pytest.approx(0.9486081, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Evaluation touches the test set only for the champion.
# ---------------------------------------------------------------------------
def test_evaluation_touches_test_only_for_champion(workspace, monkeypatch):
    _approve(workspace, ["LinearRegression", "Ridge", "Lasso"])
    trainer.train_all(
        train_path=workspace["train_path"],
        validation_path=workspace["val_path"],
    )

    # Instrument load_csv inside the evaluator to record every path it reads.
    reads: list[str] = []
    real_load = evaluator.load_csv

    def tracking_load(path, **kwargs):
        reads.append(str(path))
        return real_load(path, **kwargs)

    monkeypatch.setattr(evaluator, "load_csv", tracking_load)

    report = evaluator.evaluate(
        validation_path=workspace["val_path"],
        test_path=workspace["test_path"],
    )

    # The test split was read exactly once (only to score the champion).
    assert reads.count(str(workspace["test_path"])) == 1
    # Every candidate was compared on validation.
    assert report["n_models_compared"] == 3
    # Only the champion carries test metrics; the comparison rows do not.
    assert set(report["champion"]["test_metrics"]) >= {"rmse", "mae", "r2"}
    for row in report["validation_comparison"]:
        assert "test_metrics" not in row
