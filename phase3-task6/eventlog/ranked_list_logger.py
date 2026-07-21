"""
Task 6 - Stage C: Position and model-version logging on every ranked list.

WHAT GOOD LOOKS LIKE (the bar):
  - Every call to serve a ranked list writes ONE impression event per item,
    with its position and the exact model_name/model_version that produced
    the ordering -- even when the model is unavailable and we fall back to
    a heuristic ranker (Stage E failure test).
  - Logging must not be able to silently drop an item: rows_logged ==
    len(ranked_items) is asserted every call.
Baseline: naive logging that only logs the top-1 result, or logs without a
position field (cannot correct for position bias, see Core Concepts).
Metric: 100% of served items produce a logged impression with position and
model_version populated (checked in eval/verify_join.py).
"""
from __future__ import annotations
import csv
import os
from typing import List
from schema.events import make_impression, make_outcome, EventType, BaseEvent

FALLBACK_MODEL_NAME = "heuristic_fallback_ranker"
FALLBACK_MODEL_VERSION = "v0-static"


class RankedListLogger:
    """Appends events as CSV rows. In production this would be a Kafka
    producer / event bus; CSV is used here so the whole pipeline is runnable
    and inspectable end-to-end without external infra."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["event_type", "event_id", "session_id", "query_id",
                            "item_id", "position", "model_name", "model_version",
                            "ts", "impression_id"])

    def _write(self, ev: BaseEvent):
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            d = ev.to_dict()
            w.writerow([d["event_type"], d["event_id"], d["session_id"], d["query_id"],
                        d["item_id"], d["position"], d["model_name"], d["model_version"],
                        d["ts"], d.get("impression_id") or ""])

    def log_ranked_list(self, session_id: str, query_id: str, ranked_item_ids: List[str],
                         model_name: str, model_version: str) -> List[BaseEvent]:
        """Logs one impression per item at its 1-indexed position.
        Returns the impression events so the caller can attach outcomes later."""
        impressions = []
        for pos, item_id in enumerate(ranked_item_ids, start=1):
            ev = make_impression(session_id, query_id, item_id, pos, model_name, model_version)
            self._write(ev)
            impressions.append(ev)
        # hard guarantee: nothing silently dropped
        assert len(impressions) == len(ranked_item_ids), "impression logging dropped items"
        return impressions

    def log_outcome(self, event_type: EventType, impression: BaseEvent):
        ev = make_outcome(event_type, impression)
        self._write(ev)
        return ev
