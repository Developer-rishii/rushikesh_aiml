"""Offline (nDCG/MAP/precision@k) and online (CTR/conversion) metrics.

Offline metrics are computed per (user_id, day) impression list, ranked by
model score, evaluated against true_relevance (held-out, never used in
training). Online metrics are computed from simulated behavioural logs.
"""
import numpy as np
import pandas as pd


def _dcg(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(scored_group: pd.DataFrame, k: int = 5) -> float:
    ranked = scored_group.sort_values("score", ascending=False)
    ideal = scored_group.sort_values("true_relevance", ascending=False)
    dcg = _dcg(ranked["true_relevance"].values, k)
    idcg = _dcg(ideal["true_relevance"].values, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(scored_group: pd.DataFrame, k: int = 5, rel_threshold: float = 0.6) -> float:
    ranked = scored_group.sort_values("score", ascending=False).head(k)
    if len(ranked) == 0:
        return 0.0
    return float((ranked["true_relevance"] >= rel_threshold).mean())


def average_precision(scored_group: pd.DataFrame, rel_threshold: float = 0.6) -> float:
    ranked = scored_group.sort_values("score", ascending=False).reset_index(drop=True)
    relevant = ranked["true_relevance"] >= rel_threshold
    if relevant.sum() == 0:
        return 0.0
    hits, precisions = 0, []
    for i, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions))


def offline_report(scored_df: pd.DataFrame, group_cols=("user_id", "day"), k: int = 5) -> dict:
    """scored_df needs columns: score, true_relevance, + group_cols."""
    ndcgs, precs, aps = [], [], []
    for _, g in scored_df.groupby(list(group_cols)):
        ndcgs.append(ndcg_at_k(g, k))
        precs.append(precision_at_k(g, k))
        aps.append(average_precision(g))
    return {
        f"nDCG@{k}": round(float(np.mean(ndcgs)), 4),
        f"precision@{k}": round(float(np.mean(precs)), 4),
        "MAP": round(float(np.mean(aps)), 4),
        "n_query_groups": len(ndcgs),
    }


def online_report(events_df: pd.DataFrame) -> dict:
    """events_df needs columns: impression(1), click(0/1), application(0/1)."""
    impressions = len(events_df)
    ctr = events_df["click"].sum() / impressions if impressions else 0.0
    conversion = events_df["application"].sum() / impressions if impressions else 0.0
    return {
        "impressions": impressions,
        "CTR": round(float(ctr), 4),
        "conversion_rate": round(float(conversion), 4),
    }
