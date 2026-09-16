"""
src.model_training.factory
==========================

PURPOSE
-------
Build scikit-learn estimators from the model catalogue by name. Resolving the
``estimator_path`` string keeps the trainer decoupled from concrete imports and
lets new catalogue entries appear without editing this module.

PIPELINE POSITION
-----------------
Used by the trainer to instantiate each approved estimator (with tuned or
default hyperparameters) and by the tuning loop to build per-alpha candidates.
"""

from __future__ import annotations

import importlib
from typing import Any

from config.constants import RANDOM_STATE
from src.model_proposal.model_catalog import get_spec
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Estimators that accept a ``random_state`` argument (Lasso's coordinate descent
# can use it; LinearRegression/Ridge closed-form solvers generally do not).
# Classification adds logistic regression (liblinear) and the tree/ensemble
# families, whose fits are stochastic and must be seeded for reproducibility.
_RANDOM_STATE_ESTIMATORS = {"Lasso", "LogisticRegression"}
_RANDOM_STATE_FAMILIES = {"tree", "ensemble"}


def _resolve(estimator_path: str) -> type:
    module_name, _, class_name = estimator_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_estimator(name: str, **overrides: Any) -> Any:
    """
    Instantiate the estimator named ``name`` from the catalogue.

    Catalogue ``default_params`` are applied first, then ``overrides`` (e.g. a
    tuned ``alpha``). A ``random_state`` is injected for estimators that support
    it so runs are reproducible.
    """

    spec = get_spec(name)
    estimator_cls = _resolve(spec.estimator_path)

    params: dict[str, Any] = dict(spec.default_params)
    params.update(overrides)
    if name in _RANDOM_STATE_ESTIMATORS or spec.family in _RANDOM_STATE_FAMILIES:
        params.setdefault("random_state", RANDOM_STATE)

    estimator = estimator_cls(**params)
    logger.info("Built estimator %s(%s)", name, params)
    return estimator
