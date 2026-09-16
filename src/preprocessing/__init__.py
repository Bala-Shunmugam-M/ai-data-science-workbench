"""
src.preprocessing
=================

PURPOSE
-------
Schema-driven, leakage-safe preprocessing: deterministic cleaning, an early
stratified split, and train-only fitted transformations (median imputation,
one-hot encoding, standard scaling).

KEY INVARIANT
-------------
The split happens BEFORE any fitted transformation. Imputer, encoder, and
scaler are fit on the training split only and reused unchanged on validation
and test, so information from held-out data never leaks into the fit.
"""

from __future__ import annotations
