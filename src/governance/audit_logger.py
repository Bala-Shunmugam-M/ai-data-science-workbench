"""
src.governance.audit_logger
===========================

PURPOSE
-------
Append-only audit trail. Every governance-relevant event (proposal created,
approval granted/denied, training run, evaluation run, final selection) is
recorded as one JSON object per line (JSONL) with a UTC timestamp, the acting
actor, an event type, and a free-form payload.

PIPELINE POSITION
-----------------
Written by the proposal, approval, training, and evaluation stages. Read by the
reporting stage to summarise governance activity. The append-only JSONL format
means the trail is never rewritten, so the history is tamper-evident and
reproducible.

OUTPUTS
-------
    governance/audit/audit_log.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.constants import DEFAULT_ACTOR
from config.paths import AUDIT_LOG_PATH, ensure_dir
from src.utils.common import utc_timestamp
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def log_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    actor: str = DEFAULT_ACTOR,
    path: Path = AUDIT_LOG_PATH,
) -> dict[str, Any]:
    """
    Append one governance event to the audit trail and return the record.

    The record is flushed immediately so a crash mid-pipeline still leaves a
    durable trail. Appending (never rewriting) keeps the log tamper-evident.
    """

    record: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "actor": actor,
        "event_type": event_type,
        "payload": payload or {},
    }

    ensure_dir(path.parent)
    import json

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    logger.info("Audit event '%s' by '%s' recorded.", event_type, actor)
    return record


def read_events(path: Path = AUDIT_LOG_PATH) -> list[dict[str, Any]]:
    """Return every audit record in order, or an empty list when none exist."""

    if not path.exists():
        return []

    import json

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_events(path: Path = AUDIT_LOG_PATH) -> dict[str, Any]:
    """Return counts of total events and per-``event_type`` frequencies."""

    events = read_events(path)
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return {"total_events": len(events), "by_event_type": counts}
