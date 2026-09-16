"""
src.data_manager
================

PURPOSE
-------
Data access and governance-of-input layer: the dataset registry (reads
``config/datasets.yaml``), the semantic-schema validator, and a facade that
composes ingestion -> validation -> raw access.
"""

from __future__ import annotations
