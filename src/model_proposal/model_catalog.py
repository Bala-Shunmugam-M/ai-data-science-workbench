"""
src.model_proposal.model_catalog
================================

PURPOSE
-------
Extensible registry of candidate models. Per the team decision the catalogue is
restricted to the linear family (LinearRegression, Ridge, Lasso), but the
structure - a dict of :class:`ModelSpec` dataclasses - is deliberately generic
so ensembles or other families can be added later without any structural change
to the proposal, training, or evaluation stages.

Each entry records the sklearn class path, its hyperparameter search space, and
the governance-facing prose (strengths, limitations, interpretability notes, and
a suitability rationale for California house-price regression).

PIPELINE POSITION
-----------------
Read by the proposal stage (to build the proposal document), the training
factory (to instantiate estimators), and the trainer (to obtain search spaces).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.constants import ALPHA_GRID_SIZE


@dataclass(frozen=True)
class HyperparameterSpace:
    """
    A single tunable hyperparameter's search space.

    ``scale`` is either ``"log"`` (log-spaced grid between ``low`` and ``high``)
    or ``"fixed"`` (a single value in ``low``). ``num`` grid points are swept on
    the validation split during tuning.
    """

    name: str
    scale: str  # "log" (log-spaced), "int" (integer grid), or "fixed"
    low: float
    high: float
    num: int = ALPHA_GRID_SIZE

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "low": self.low,
            "high": self.high,
            "num": self.num,
        }


@dataclass(frozen=True)
class ModelSpec:
    """One catalogue entry: how to build the estimator and why it is a candidate."""

    name: str
    estimator_path: str
    family: str
    search_space: dict[str, HyperparameterSpace]
    default_params: dict[str, Any]
    tunable: bool
    strengths: list[str]
    limitations: list[str]
    interpretability: str
    suitability: str
    task: str = "regression"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimator_path": self.estimator_path,
            "family": self.family,
            "task": self.task,
            "search_space": {k: v.as_dict() for k, v in self.search_space.items()},
            "default_params": self.default_params,
            "tunable": self.tunable,
            "strengths": self.strengths,
            "limitations": self.limitations,
            "interpretability": self.interpretability,
            "suitability": self.suitability,
        }


# ---------------------------------------------------------------------------
# The catalogue. Keyed by model name; add new families here without touching
# downstream stages.
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, ModelSpec] = {
    "LinearRegression": ModelSpec(
        name="LinearRegression",
        estimator_path="sklearn.linear_model.LinearRegression",
        family="linear",
        search_space={},  # No regularisation to tune.
        default_params={},
        tunable=False,
        strengths=[
            "Fully transparent: one coefficient per predictor, no hyperparameters.",
            "Fast to fit and score; an interpretable managerial baseline.",
            "Coefficients read directly as marginal dollar effects.",
        ],
        limitations=[
            "No regularisation, so it is sensitive to the strong multicollinearity "
            "among the room/bedroom/population/household counts.",
            "Cannot capture the coastal/geographic nonlinearity flagged in EDA.",
            "Coefficients can become unstable when predictors are near-collinear.",
        ],
        interpretability="Highest: each standardized coefficient is a direct, "
        "signed feature effect with no shrinkage to explain away.",
        suitability="A defensible, transparent baseline for house-price "
        "regression and the reference point every other model must beat.",
    ),
    "Ridge": ModelSpec(
        name="Ridge",
        estimator_path="sklearn.linear_model.Ridge",
        family="linear",
        search_space={
            "alpha": HyperparameterSpace("alpha", "log", 1e-3, 1e3),
        },
        default_params={},
        tunable=True,
        strengths=[
            "L2 shrinkage stabilises coefficients under the heavy multicollinearity "
            "present in the housing counts.",
            "Keeps every predictor while damping variance; robust, well-conditioned.",
            "Single interpretable regularisation strength (alpha) tuned on validation.",
        ],
        limitations=[
            "Does not zero out predictors, so it offers no feature selection.",
            "Still a linear model: no interaction or curvature unless engineered.",
            "Coefficients are shrunk, so raw magnitudes understate isolated effects.",
        ],
        interpretability="High: coefficients remain signed and comparable after "
        "standardization; shrinkage is a single, explainable knob.",
        suitability="Strong default for this collinear predictor set; expected to "
        "improve coefficient stability over plain LinearRegression.",
    ),
    "Lasso": ModelSpec(
        name="Lasso",
        estimator_path="sklearn.linear_model.Lasso",
        family="linear",
        search_space={
            "alpha": HyperparameterSpace("alpha", "log", 1e-4, 1e1),
        },
        # tol is set relative to the dollar-scale target: the median house value
        # is ~1e5, so a coordinate-descent duality-gap tolerance of 0.1 is far
        # below any meaningful cent-level precision yet lets Lasso converge in
        # seconds. Empirically it yields the identical solution to the default
        # tol at a fraction of the run time (see docs / final report).
        default_params={"max_iter": 10000, "tol": 0.1},
        tunable=True,
        strengths=[
            "L1 penalty performs automatic feature selection, zeroing weak "
            "predictors for a sparse, easily-communicated model.",
            "Useful for isolating the handful of true value drivers.",
            "Single interpretable regularisation strength (alpha) tuned on validation.",
        ],
        limitations=[
            "Among a group of collinear predictors it keeps one somewhat "
            "arbitrarily, which can complicate causal reading.",
            "Can be numerically slow to converge at very small alpha (max_iter raised).",
            "Still linear: no curvature or interactions unless engineered.",
        ],
        interpretability="High and sparse: the surviving non-zero coefficients "
        "are the model's shortlist of value drivers for the managerial report.",
        suitability="Complements Ridge by producing a parsimonious driver list, "
        "aiding the executive briefing's 'top drivers' narrative.",
    ),
    # -----------------------------------------------------------------------
    # Classification family (churn project). Each has exactly one tunable
    # hyperparameter so the trainer's single-parameter validation sweep serves
    # both tasks unchanged.
    # -----------------------------------------------------------------------
    "LogisticRegression": ModelSpec(
        name="LogisticRegression",
        estimator_path="sklearn.linear_model.LogisticRegression",
        family="linear",
        task="classification",
        search_space={
            # Inverse regularisation strength; smaller C = stronger shrinkage.
            "C": HyperparameterSpace("C", "log", 1e-2, 1e2),
        },
        default_params={"max_iter": 2000, "solver": "liblinear"},
        tunable=True,
        strengths=[
            "Fully interpretable: each coefficient is a log-odds effect, and "
            "exp(coef) reads as an odds ratio for the churn briefing.",
            "Well-calibrated probabilities, ideal for the retention simulator.",
            "Single regularisation knob (C) tuned on the validation split.",
        ],
        limitations=[
            "Linear decision boundary; cannot capture interactions unless engineered.",
            "Sensitive to strong multicollinearity among predictors.",
        ],
        interpretability="Highest among the churn models: signed coefficients "
        "convert directly to odds ratios per driver.",
        suitability="The interpretable baseline for churn: quantifies which "
        "factors drive churn after controlling for the others.",
    ),
    "DecisionTreeClassifier": ModelSpec(
        name="DecisionTreeClassifier",
        estimator_path="sklearn.tree.DecisionTreeClassifier",
        family="tree",
        task="classification",
        search_space={
            "max_depth": HyperparameterSpace("max_depth", "int", 2, 12),
        },
        default_params={"class_weight": "balanced"},
        tunable=True,
        strengths=[
            "Produces human-readable if-then rules the retention team can act on.",
            "Captures non-linear splits and interactions automatically.",
            "class_weight='balanced' counters the churn class imbalance.",
        ],
        limitations=[
            "A single tree can overfit; depth is capped and tuned on validation.",
            "Less stable than an ensemble; small data changes can reshape splits.",
        ],
        interpretability="Very high: the fitted tree is a flowchart of decision "
        "rules; feature_importances_ rank the drivers.",
        suitability="The interpretable non-linear model for churn: gives advisors "
        "concrete rules ('month-to-month + high charges -> high risk').",
    ),
    "RandomForestClassifier": ModelSpec(
        name="RandomForestClassifier",
        estimator_path="sklearn.ensemble.RandomForestClassifier",
        family="ensemble",
        task="classification",
        search_space={
            "max_depth": HyperparameterSpace("max_depth", "int", 4, 16),
        },
        default_params={"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1},
        tunable=True,
        strengths=[
            "Strongest predictive accuracy of the churn models via bagged trees.",
            "Robust to noise and multicollinearity; low-variance predictions.",
            "permutation/impurity feature importances still explain the drivers.",
        ],
        limitations=[
            "Not a single readable rule set; interpret via importances, not a tree.",
            "Heavier to train and score than the linear/tree baselines.",
        ],
        interpretability="Moderate: no single flowchart, but feature_importances_ "
        "and partial effects rank the churn drivers.",
        suitability="The accuracy benchmark for churn every simpler model is "
        "measured against; powers the retention simulator's risk scores.",
    ),
}


def list_model_names() -> list[str]:
    """Return catalogue model names in registration order."""

    return list(MODEL_CATALOG.keys())


def models_for_task(task: str) -> list[str]:
    """Return catalogue model names whose ``task`` matches ``task``."""

    return [name for name, spec in MODEL_CATALOG.items() if spec.task == task]


def get_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` or raise ``KeyError``."""

    if name not in MODEL_CATALOG:
        raise KeyError(
            f"Unknown model '{name}'. Known models: {list_model_names()}"
        )
    return MODEL_CATALOG[name]


def catalog_as_dicts() -> list[dict[str, Any]]:
    """Return the full catalogue as JSON-serialisable dicts."""

    return [spec.as_dict() for spec in MODEL_CATALOG.values()]
