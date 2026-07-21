"""
The inference path being profiled. Two server classes share the same model
and feature-store interface so the BEFORE/AFTER comparison is apples-to-apples
-- only the serving *strategy* changes, not the underlying data.

BaselineServer  (BEFORE): per-candidate feature fetch (one round trip each),
                          full-size model, no caching.
OptimizedServer (AFTER):  batched feature fetch (one round trip per page),
                          distilled small model, LRU score cache for
                          candidates re-scored within the TTL window, and a
                          designed-degradation fallback if the model is
                          unavailable.
"""
import time
import numpy as np
from collections import OrderedDict

from src.config import ALL_FEATURES, CHEAP_FEATURES
from src.data_pipeline import compute_cheap_features, to_feature_matrix


class LRUScoreCache:
    """Reuses scores that will not change within a short TTL -- the
    'caching & batching' concept from the guide's core-concepts section."""

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self._store = OrderedDict()

    def get(self, key):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key, value):
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def __len__(self):
        return len(self._store)


class ModelUnavailable(Exception):
    pass


class BaselineServer:
    """BEFORE: naive per-request serving path."""

    def __init__(self, model, feature_store):
        self.model = model
        self.feature_store = feature_store

    def rank(self, job_id, candidate_rows):
        rows = []
        for r in candidate_rows:
            cheap = compute_cheap_features(r)
            fetched = self.feature_store.fetch_one(r["candidate_id"])  # 1 RTT per candidate
            rows.append({**cheap, **fetched})
        X = to_feature_matrix(rows, ALL_FEATURES)
        scores = self.model.predict(X)
        return scores


class OptimizedServer:
    """AFTER: batched fetch + cache + small distilled model + fallback."""

    def __init__(self, model, feature_store, cache: LRUScoreCache = None,
                 fallback_scorer=None, force_unavailable=False):
        self.model = model
        self.feature_store = feature_store
        self.cache = cache if cache is not None else LRUScoreCache()
        self.fallback_scorer = fallback_scorer
        self.force_unavailable = force_unavailable  # for failure-injection tests

    def rank(self, job_id, candidate_rows, cache_key_fn=None):
        try:
            if self.force_unavailable:
                raise ModelUnavailable("model service down (injected failure)")

            cids = [r["candidate_id"] for r in candidate_rows]
            fetched_map = self.feature_store.fetch_batch(cids)  # 1 RTT for whole page

            to_score_idx, to_score_rows, scores = [], [], [None] * len(candidate_rows)
            for i, r in enumerate(candidate_rows):
                key = cache_key_fn(r) if cache_key_fn else None
                cached = self.cache.get(key) if key is not None else None
                if cached is not None:
                    scores[i] = cached
                else:
                    cheap = compute_cheap_features(r)
                    row = {**cheap, **fetched_map[r["candidate_id"]]}
                    to_score_idx.append(i)
                    to_score_rows.append((key, row))

            if to_score_rows:
                X = to_feature_matrix([row for _, row in to_score_rows], ALL_FEATURES)
                preds = self.model.predict(X)
                for (key, _), i, p in zip(to_score_rows, to_score_idx, preds):
                    scores[i] = float(p)
                    if key is not None:
                        self.cache.put(key, float(p))
            return np.array(scores), "model"
        except ModelUnavailable:
            # Designed degradation: fall back to a cheap non-ML popularity
            # heuristic instead of failing the request outright.
            if self.fallback_scorer is None:
                raise
            return self.fallback_scorer(candidate_rows), "fallback"


def popularity_fallback_scorer(candidate_rows):
    """Cheap, model-free heuristic used only when the model is unavailable.
    Not accurate, but keeps the endpoint alive and ranks by a reasonable
    proxy (activity + response rate) instead of 500-ing."""
    return np.array([
        0.6 * r["candidate_activity_score"] + 0.4 * r["candidate_past_response_rate"]
        for r in candidate_rows
    ])
