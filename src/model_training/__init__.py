"""
src.model_training
==================

PURPOSE
-------
Stage 7: Model Training. Builds approved estimators, prepares a leakage-safe
modelling matrix (impute/encode/scale fit on train only), tunes Ridge/Lasso on
the validation split, and persists versioned, registered models.

PIPELINE POSITION
-----------------
    approval -> [training] -> evaluation
"""

from __future__ import annotations
