"""
Offline ranking metrics computed per job_id (query) and averaged -- this is
the "held-out data you did not tune on" evaluation the guide requires.
"""
import numpy as np
import pandas as pd


def _dcg(rels, k):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) / discounts))


def ndcg_at_k(true_rel_sorted_by_pred, k):
    ideal = sorted(true_rel_sorted_by_pred, reverse=True)
    dcg = _dcg(true_rel_sorted_by_pred, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(true_rel_sorted_by_pred, k, positive_thresh=1):
    top = true_rel_sorted_by_pred[:k]
    if len(top) == 0:
        return 0.0
    return float(np.mean([1 if r >= positive_thresh else 0 for r in top]))


def average_precision_at_k(true_rel_sorted_by_pred, k, positive_thresh=1):
    top = true_rel_sorted_by_pred[:k]
    hits, ap_sum = 0, 0.0
    for i, r in enumerate(top, start=1):
        if r >= positive_thresh:
            hits += 1
            ap_sum += hits / i
    return ap_sum / hits if hits > 0 else 0.0


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str, job_col: str, k=10):
    """df must contain job_col, label_col, score_col for a held-out set."""
    ndcgs, precs, aps = [], [], []
    for job_id, g in df.groupby(job_col):
        g_sorted = g.sort_values(score_col, ascending=False)
        rels = g_sorted[label_col].tolist()
        ndcgs.append(ndcg_at_k(rels, k))
        precs.append(precision_at_k(rels, k))
        aps.append(average_precision_at_k(rels, k))
    return {
        f"nDCG@{k}": float(np.mean(ndcgs)),
        f"P@{k}": float(np.mean(precs)),
        f"MAP@{k}": float(np.mean(aps)),
        "n_queries": int(df[job_col].nunique()),
    }
