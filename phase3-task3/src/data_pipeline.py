"""
Feature computation. This module is intentionally the SAME import used by
both training (`train_baseline.py` / `train_optimized.py`) and serving
(`serving.py`), because train/serve skew is caused by having two different
implementations of "the same" feature. Here there is only one.

`FeatureStore` simulates the network round-trip cost of fetching the four
FETCHED_FEATURES from a real feature store, which is what the latency
profiler will show is the actual bottleneck (see profiling.py).
"""
import time
import numpy as np
import pandas as pd
from src.config import CHEAP_FEATURES, FETCHED_FEATURES, ALL_FEATURES


class FeatureStore:
    """Stand-in for a network feature store (e.g. Redis/DynamoDB-backed)."""

    def __init__(self, df: pd.DataFrame, per_call_latency_ms=0.9, batch_overhead_ms=1.2):
        # Index by candidate_id for O(1) lookup, as a real feature store would be keyed.
        self._store = df.set_index("candidate_id")[FETCHED_FEATURES].to_dict(orient="index")
        self.per_call_latency_ms = per_call_latency_ms   # simulated network RTT per item
        self.batch_overhead_ms = batch_overhead_ms         # fixed cost per batch call

    def fetch_one(self, candidate_id: str) -> dict:
        """Unbatched fetch: one network round trip per candidate. This is the
        naive path (used by the BEFORE / baseline server)."""
        time.sleep(self.per_call_latency_ms / 1000.0)
        return dict(self._store[candidate_id])

    def fetch_batch(self, candidate_ids: list) -> dict:
        """Batched fetch: one round trip for the whole page of candidates.
        Used by the AFTER / optimized server."""
        time.sleep(self.batch_overhead_ms / 1000.0)
        return {cid: dict(self._store[cid]) for cid in candidate_ids}


def compute_cheap_features(raw_record: dict) -> dict:
    """Features already present on the incoming request -- zero I/O cost.
    This is the single implementation used at both train and serve time."""
    return {k: raw_record[k] for k in CHEAP_FEATURES}


def to_feature_matrix(rows: list, feature_order=ALL_FEATURES) -> np.ndarray:
    return np.array([[r[f] for f in feature_order] for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Train/serve skew demo (used by src/skew_check.py)
# ---------------------------------------------------------------------------
def compute_cheap_features_BUGGY(raw_record: dict) -> dict:
    """A second, *divergent* implementation of the same cheap features, as if
    a serving engineer had re-implemented the feature logic instead of
    importing it. `distance_km` is accidentally left in miles (no unit
    conversion) -- a classic train/serve skew bug. Used ONLY by
    skew_check.py to demonstrate detection; never used by real serving."""
    out = dict(raw_record)
    miles = raw_record["distance_km"] / 1.60934  # bug: forgot to convert back
    return {
        "skill_overlap": raw_record["skill_overlap"],
        "years_experience": raw_record["years_experience"],
        "distance_km": miles,
        "salary_gap_pct": raw_record["salary_gap_pct"],
    }
