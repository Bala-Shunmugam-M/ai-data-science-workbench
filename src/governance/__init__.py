"""
src.governance
==============

PURPOSE
-------
Cross-cutting governance layer for the workbench: append-only audit trail,
approval workflow (features + models), data/model lineage, and a human-readable
decision registry.

PIPELINE POSITION
-----------------
Governance is consulted or written by every Milestone-2 stage: the proposal
stage records decisions, the approval workflow gates training, the trainer and
evaluator audit-log and lineage-log their runs, and the reporter summarises the
governance state.
"""

from __future__ import annotations


class GovernanceError(RuntimeError):
    """
    Raised when a governance policy is violated.

    The canonical use is the training guard: an estimator that is not present in
    ``governance/approvals/model_approval.json`` must never be trained.
    """
