"""
src.artifacts
=============

PURPOSE
-------
Versioned, content-addressable artifact management: a content-hash store for
arbitrary files, the model registry (``models/model_registry.json``), and the
append-only experiment tracker (``artifacts/experiments/experiments.jsonl``).

PIPELINE POSITION
-----------------
Written by the training and evaluation stages; read by evaluation, reporting,
and the lineage tracker.
"""

from __future__ import annotations
