"""
src.model_proposal
==================

PURPOSE
-------
Stage 6: Model Proposal. A governed catalogue of candidate estimators (linear
family per team decision) and a data-aware proposal document that recommends
which to train and how to validate them.

PIPELINE POSITION
-----------------
    feature engineering -> [model proposal] -> approval -> training
"""

from __future__ import annotations
