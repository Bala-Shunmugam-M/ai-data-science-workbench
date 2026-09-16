"""
config
======

PURPOSE
-------
Central configuration package for the AI Data Science Workbench.

This package is the single source of truth for:

- filesystem locations (``config.paths``);
- reproducibility constants and pipeline parameters (``config.constants``);
- declarative settings (``settings.yaml``, ``datasets.yaml``).

No stage-specific logic lives here; modules under ``src`` import from this
package so that directory conventions and thresholds are never hard-coded in
more than one place.
"""

from __future__ import annotations
