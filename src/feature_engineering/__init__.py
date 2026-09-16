"""
src.feature_engineering
=======================

PURPOSE
-------
Governed, two-step feature engineering:

1. ``feature_proposal`` writes a formal plan (what, why, source variables,
   automatic/optional, validation required, implementing script) to governance
   approvals and a results CSV. It creates no features.
2. ``transformers`` + ``pipeline`` execute ONLY the features marked
   ``automatic=True`` (approved deterministic ratios and log transforms),
   producing the engineered train/validation/test datasets.
"""

from __future__ import annotations
