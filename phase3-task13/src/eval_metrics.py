"""
Stage C step 3 / Section 3 prerequisite: offline metrics (nDCG, MAP,
precision@k). Implemented from scratch since scikit-learn's ranking
metrics module was not reliably importable offline for graded relevance.
"""
from __future__ import annotations
import math


def precision_at_k(ranked_ids: list[str], rel_map: dict[str, int], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rel_map.get(rid, 0) > 0)
    return hits / len(top)


def dcg_at_k(ranked_ids: list[str], rel_map: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k]):
        rel = rel_map.get(rid, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids: list[str], rel_map: dict[str, int], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, rel_map, k)
    ideal_order = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked_ids: list[str], rel_map: dict[str, int]) -> float:
    n_rel = sum(1 for v in rel_map.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, rid in enumerate(ranked_ids):
        if rel_map.get(rid, 0) > 0:
            hits += 1
            precisions.append(hits / (i + 1))
    return sum(precisions) / n_rel if precisions else 0.0


def evaluate_run(all_rankings: dict[str, list[str]], all_rel: dict[str, dict[str, int]], k: int = 10) -> dict:
    """all_rankings: {query_id: [ranked resume_id,...]}, all_rel: {query_id: {resume_id: relevance}}"""
    ndcgs, maps, precs = [], [], []
    for qid, ranked in all_rankings.items():
        rel_map = all_rel.get(qid, {})
        ndcgs.append(ndcg_at_k(ranked, rel_map, k))
        maps.append(average_precision(ranked, rel_map))
        precs.append(precision_at_k(ranked, rel_map, k))
    n = max(len(ndcgs), 1)
    return {
        f"nDCG@{k}": sum(ndcgs) / n,
        "MAP": sum(maps) / n,
        f"precision@{k}": sum(precs) / n,
        "n_queries": len(ndcgs),
    }
