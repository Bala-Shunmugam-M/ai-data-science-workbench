"""
config.constants
================

PURPOSE
-------
Reproducibility constants and pipeline parameters shared across stages.

PIPELINE POSITION
-----------------
Imported by the splitter, imputer, encoder, scaler, and feature-engineering
modules. These values reproduce the professor's starter conventions exactly
and must not be redefined elsewhere.
"""

from __future__ import annotations

# Global reproducibility seed (professor's split.py uses 42 throughout).
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Stratified-split configuration.
#
# ``median_income`` is binned into an ``income_category`` used only to stratify
# the split; the temporary column is dropped from the saved splits. Bins and
# fractions reproduce the professor's split.py exactly.
# ---------------------------------------------------------------------------
INCOME_BINS: list[float] = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
INCOME_LABELS: list[int] = [1, 2, 3, 4, 5]

TRAIN_FRACTION: float = 0.70
VALIDATION_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15

# Derived two-step split sizes (train vs temp, then validation vs test).
# temp = validation + test = 0.30 of the data; within temp the validation/test
# halves are 0.50 each, yielding the 0.70 / 0.15 / 0.15 global split.
TEMP_TEST_SIZE: float = round(VALIDATION_FRACTION + TEST_FRACTION, 10)  # 0.30
VALIDATION_TEST_SPLIT: float = round(
    TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION), 10
)  # 0.50

# ---------------------------------------------------------------------------
# EDA thresholds (carried over from the professor's eda / analysis scripts).
# ---------------------------------------------------------------------------
IQR_MULTIPLIER: float = 1.5
MODIFIED_Z_THRESHOLD: float = 3.5
HIGH_PAIRWISE_CORRELATION: float = 0.80
MATERIAL_OUTLIER_PERCENTAGE: float = 5.0

# LOWESS / curvature-screen parameters (professor's analysis summary).
LOWESS_SAMPLE_SIZE: int = 5000
LOWESS_FRACTION: float = 0.25
MILD_CURVATURE_THRESHOLD: float = 0.08
MODERATE_CURVATURE_THRESHOLD: float = 0.16
STRONG_CURVATURE_THRESHOLD: float = 0.28

# ---------------------------------------------------------------------------
# Milestone 2: modelling / governance parameters.
# ---------------------------------------------------------------------------
# Prediction target for the California Housing regression problem.
TARGET_COLUMN: str = "median_house_value"

# Default governance actor recorded on audit and approval events.
DEFAULT_ACTOR: str = "analyst"

# Number of candidate alpha values swept when tuning Ridge / Lasso on the
# validation split (log-spaced within each model's catalog search space).
ALPHA_GRID_SIZE: int = 25

# The regression metric used to rank models on the validation split and to
# choose the champion for the single untouched-test evaluation.
SELECTION_METRIC: str = "rmse"

# California Housing target is capped at $500,001 in the source census data.
TARGET_VALUE_CAP: float = 500_001.0
