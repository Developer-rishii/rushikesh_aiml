"""
Task 6 - Stage B: Event schema for ranked results.

WHAT GOOD LOOKS LIKE (the bar):
  - Every ranked list shown to a candidate produces one IMPRESSION event per
    item, containing enough fields to reconstruct exactly what was shown,
    in what order, by which model.
  - Every downstream action (click / apply / shortlist) is a separate event
    that carries the SAME impression_id so it can be joined back.
  - Nothing is logged without: session_id, query_id, item_id, position,
    model_name, model_version, ts.

Baseline we must beat: "no schema" (current state = ranking results are
served but nothing about position or model identity is captured, so no
online metric can ever be computed). Any schema that lets us join
impression -> outcome and reconstruct order beats that baseline.
Metric that decides it: 100% join-ability of outcome events back to their
impression (see eval/verify_join.py) and 0 missing (position, model_version)
fields on required events.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import uuid
import time


class EventType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"
    APPLY = "apply"
    SHORTLIST = "shortlist"


@dataclass
class BaseEvent:
    event_type: EventType
    event_id: str
    session_id: str
    query_id: str          # identifies the ranking request (one search / feed load)
    item_id: str            # candidate or job id being ranked
    position: int            # 1-indexed rank position shown to the user (Core concept: position bias)
    model_name: str          # e.g. "ltr_ranker"
    model_version: str        # e.g. "2026-07-21T10:00Z" or git sha / mlflow run id
    ts: float                 # unix epoch seconds
    impression_id: Optional[str] = None  # set on click/apply/shortlist, points back to the impression event_id

    def to_dict(self):
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


def make_impression(session_id: str, query_id: str, item_id: str, position: int,
                     model_name: str, model_version: str) -> BaseEvent:
    return BaseEvent(
        event_type=EventType.IMPRESSION,
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        query_id=query_id,
        item_id=item_id,
        position=position,
        model_name=model_name,
        model_version=model_version,
        ts=time.time(),
    )


def make_outcome(event_type: EventType, impression: BaseEvent) -> BaseEvent:
    """Outcome events (click/apply/shortlist) inherit identity fields from the
    impression they resulted from, and carry impression_id for the join."""
    assert event_type != EventType.IMPRESSION
    return BaseEvent(
        event_type=event_type,
        event_id=str(uuid.uuid4()),
        session_id=impression.session_id,
        query_id=impression.query_id,
        item_id=impression.item_id,
        position=impression.position,
        model_name=impression.model_name,
        model_version=impression.model_version,
        ts=time.time(),
        impression_id=impression.event_id,
    )


REQUIRED_FIELDS = ["event_type", "event_id", "session_id", "query_id",
                    "item_id", "position", "model_name", "model_version", "ts"]


def validate_event(d: dict) -> list[str]:
    """Returns a list of validation errors (empty = valid)."""
    errors = []
    for f in REQUIRED_FIELDS:
        if d.get(f) in (None, ""):
            errors.append(f"missing required field: {f}")
    if d.get("event_type") != "impression" and not d.get("impression_id"):
        errors.append("outcome event missing impression_id (cannot be joined)")
    return errors
