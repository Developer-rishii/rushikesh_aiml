"""
evaluate.py
===========
Offline ranking metrics computed per job_id group (each job's candidate
pool is one ranked list), matching how PlaceMux actually serves rankings.
"""
import numpy as np
import pandas as pd


def _dcg(rels):
    rels = np.asarray(rels, dtype=float)
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return np.sum(rels / discounts)


def ndcg_at_k(y_true, y_score, groups, k=10):
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    scores = []
    for _, grp in df.groupby("g"):
        grp = grp.sort_values("s", ascending=False).head(k)
        ideal = grp.sort_values("y", ascending=False)
        dcg = _dcg(grp["y"].values)
        idcg = _dcg(ideal["y"].values)
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(y_true, y_score, groups, k=10):
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    aps = []
    for _, grp in df.groupby("g"):
        grp = grp.sort_values("s", ascending=False).head(k)
        y = grp["y"].values
        if y.sum() == 0:
            continue
        hits, prec_sum = 0, 0.0
        for i, rel in enumerate(y, start=1):
            if rel:
                hits += 1
                prec_sum += hits / i
        aps.append(prec_sum / y.sum())
    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(y_true, y_score, groups, k=5):
    df = pd.DataFrame({"y": y_true, "s": y_score, "g": groups})
    precs = []
    for _, grp in df.groupby("g"):
        grp = grp.sort_values("s", ascending=False).head(k)
        if len(grp) == 0:
            continue
        precs.append(grp["y"].mean())
    return float(np.mean(precs)) if precs else 0.0


def offline_report(y_true, y_score, groups, k=10):
    return {
        "ndcg@10": round(ndcg_at_k(y_true, y_score, groups, 10), 4),
        "map@10": round(map_at_k(y_true, y_score, groups, 10), 4),
        "precision@5": round(precision_at_k(y_true, y_score, groups, 5), 4),
    }
