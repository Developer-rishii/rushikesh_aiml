"""
Stage D: Hybrid retrieval (semantic + keyword) with tuned weighting.

Both score lists are min-max normalized per query before combining
(semantic cosine and BM25 scores are on incompatible scales, and
combining them raw silently lets whichever has the bigger numeric range
dominate - a subtle bug worth documenting).
"""
from __future__ import annotations
import numpy as np


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_search(semantic_scores: dict[str, float], bm25_scores: dict[str, float],
                   alpha: float, top_k: int = 10) -> list[tuple[str, float]]:
    """alpha=1.0 -> pure semantic, alpha=0.0 -> pure keyword."""
    sem_n = _minmax(semantic_scores)
    bm25_n = _minmax(bm25_scores)
    all_ids = set(sem_n) | set(bm25_n)
    combined = {rid: alpha * sem_n.get(rid, 0.0) + (1 - alpha) * bm25_n.get(rid, 0.0) for rid in all_ids}
    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return ranked[:top_k]


def tune_alpha(dev_semantic: dict, dev_bm25: dict, dev_rel: dict, k: int = 10, grid=None):
    """Grid-search alpha on the DEV split only (never on the held-out test split -
    this is what keeps the final reported numbers honest, per Pitfall #1)."""
    from .eval_metrics import evaluate_run
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.0, 1.05, 0.05)]
    best_alpha, best_score, trace = None, -1.0, []
    for alpha in grid:
        rankings = {}
        for qid in dev_rel:
            ranked = hybrid_search(dev_semantic[qid], dev_bm25[qid], alpha, top_k=k)
            rankings[qid] = [rid for rid, _ in ranked]
        res = evaluate_run(rankings, dev_rel, k=k)
        trace.append({"alpha": alpha, **res})
        if res[f"nDCG@{k}"] > best_score:
            best_score, best_alpha = res[f"nDCG@{k}"], alpha
    return best_alpha, best_score, trace
