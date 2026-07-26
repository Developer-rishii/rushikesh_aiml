"""metrics.py -- nDCG@k, MAP@k, Precision@k for graded/binary relevance."""
import numpy as np


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2 ** rels - 1) / discounts))


def ndcg_at_k(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def average_precision_at_k(rel_binary, k):
    rel_binary = np.asarray(rel_binary)[:k]
    if rel_binary.sum() == 0:
        return 0.0
    hits, precisions = 0, []
    for i, r in enumerate(rel_binary, start=1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return float(np.sum(precisions) / rel_binary.sum())


def precision_at_k(rel_binary, k):
    rel_binary = np.asarray(rel_binary)[:k]
    if rel_binary.size == 0:
        return 0.0
    return float(rel_binary.mean())


def evaluate_ranking(df, score_col, relevance_col="true_relevance", k=10, group_col="job_id"):
    """df must contain one row per (job, candidate). Ranks candidates within
    each job by score_col (desc) then computes nDCG@k / MAP@k / P@5 using
    relevance_col as graded truth. Binary relevance for MAP/P@k = top-30% of
    relevance within the job (i.e. "genuinely good matches")."""
    ndcgs, maps, precs = [], [], []
    for job_id, g in df.groupby(group_col):
        g = g.sort_values(score_col, ascending=False)
        rels = g[relevance_col].values
        thresh = np.quantile(rels, 0.70)
        rel_bin = (rels >= thresh).astype(int)
        ndcgs.append(ndcg_at_k(rels, k))
        maps.append(average_precision_at_k(rel_bin, k))
        precs.append(precision_at_k(rel_bin, 5))
    return {
        f"nDCG@{k}": float(np.mean(ndcgs)),
        f"MAP@{k}": float(np.mean(maps)),
        "Precision@5": float(np.mean(precs)),
    }
