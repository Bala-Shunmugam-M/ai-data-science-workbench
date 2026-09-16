"""
src.pipelines
=============

PURPOSE
-------
Stage-level orchestration. Each pipeline exposes a ``run()`` that logs its
progress, calls the domain modules, and reads/writes the canonical files under
``data/`` and ``results/``. The ``main.py`` CLI subcommands call these.
"""

from __future__ import annotations
