"""
app
===

PURPOSE
-------
Streamlit GUI package for the AI Data Science Workbench (Milestone 3). A
READ-ONLY dashboard over the artifacts every pipeline stage already writes to
``data/``, ``results/``, ``artifacts/``, ``governance/``, and ``models/``. The
only exception is the "Run stage" section on the Pipeline page, which shells
out to ``python main.py pipeline ...`` rather than mutating state in-process.
"""

from __future__ import annotations
