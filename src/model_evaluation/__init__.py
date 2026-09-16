"""
src.model_evaluation
====================

PURPOSE
-------
Stage 8: Model Evaluation. Compares every registered model on the validation
split, promotes the best (champion), and evaluates ONLY the champion once on the
untouched test split, with residual diagnostics and a plain-English selection
narrative.

PIPELINE POSITION
-----------------
    training -> [evaluation] -> explainability -> reporting
"""

from __future__ import annotations
