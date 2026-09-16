"""
src.reporting
=============

PURPOSE
-------
Assemble the single self-contained project report. ``report_generator`` gathers
every stage's artifacts into a plain data model; ``html_reporter`` renders that
model to a print-friendly, base64-embedded HTML file with minimal inline CSS.

PIPELINE POSITION
-----------------
    explainability -> [reporting] -> deliverable

OUTPUTS
-------
    artifacts/reports/project_report.html
"""

from __future__ import annotations
