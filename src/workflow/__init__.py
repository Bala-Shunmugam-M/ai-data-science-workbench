"""
src.workflow
============

PURPOSE
-------
Milestone-3 orchestration layer: a declarative stage registry
(:mod:`src.workflow.pipeline_builder`) and a light DAG runner
(:mod:`src.workflow.orchestrator`) over the existing pipeline ``run()``
functions in :mod:`src.pipelines`. No stage behavior is changed here - this
package only sequences, dependency-checks, and status-tracks the stages that
already exist.
"""

from __future__ import annotations
