"""
src.eda
=======

PURPOSE
-------
Exploratory analysis of the TRAINING split only (the test set stays untouched):

- ``understanding``  : schema-aware variable summary + data-quality checks;
- ``explorer``       : distributions, correlations, categorical breakdowns,
                       outlier screens, and matplotlib figures;
- ``analysis_summary``: candidate-variable narrative and modelling guidance.

All figures are written with ``savefig`` (never ``show``).
"""

from __future__ import annotations
