"""
offline_eval.py
----------------
Evaluate baseline vs treatment on HELD-OUT queries only (never tuned on).
Reports nDCG@10, MAP, precision@5 — the offline metrics named in the study
guide's prerequisites — and is explicit that an offline win is not proof of
an online win (that's what the A/B test is for).
"""

import json

import numpy as np
import pandas as pd

from train_ranker import baseline_score_fn, TreatmentRanker, train


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2**rels - 1) * discounts))


def ndcg_at_k(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def average_precision(rels, threshold=2):
    """Binary relevance = 1 if graded relevance >= threshold."""
    binary = [1 if r >= threshold else 0 for r in rels]
    hits, s = 0, 0.0
    for i, r in enumerate(binary, start=1):
        if r:
            hits += 1
            s += hits / i
    return s / hits if hits else 0.0


def precision_at_k(rels, k, threshold=2):
    top = rels[:k]
    if len(top) == 0:
        return 0.0
    return sum(1 for r in top if r >= threshold) / len(top)


def rank_query(df_q, score_fn, as_of_day):
    scores = score_fn(df_q, as_of_day=as_of_day)
    order = np.argsort(-scores)
    return df_q.iloc[order]["relevance"].to_numpy()


def evaluate(test_df: pd.DataFrame, treatment: TreatmentRanker, as_of_day: int):
    metrics = {"baseline": {"ndcg": [], "ap": [], "p5": []},
               "treatment": {"ndcg": [], "ap": [], "p5": []}}
    for qid, group in test_df.groupby("query_id"):
        for name, fn in [("baseline", baseline_score_fn), ("treatment", treatment.score)]:
            rels = rank_query(group, fn, as_of_day)
            metrics[name]["ndcg"].append(ndcg_at_k(rels, 10))
            metrics[name]["ap"].append(average_precision(rels))
            metrics[name]["p5"].append(precision_at_k(rels, 5))

    report = {}
    for name in ("baseline", "treatment"):
        report[name] = {
            "nDCG@10": float(np.mean(metrics[name]["ndcg"])),
            "MAP": float(np.mean(metrics[name]["ap"])),
            "precision@5": float(np.mean(metrics[name]["p5"])),
            "n_queries": int(test_df["query_id"].nunique()),
        }
    report["gap_treatment_minus_baseline"] = {
        k: report["treatment"][k] - report["baseline"][k]
        for k in ("nDCG@10", "MAP", "precision@5")
    }
    report["note"] = (
        "Offline win here is necessary but not sufficient: nDCG/MAP/precision "
        "improve on held-out relevance labels, but relevance labels are a proxy "
        "for what recruiters actually do (click/apply/shortlist). The A/B test "
        "(ab_simulation.py + readout.py) is what determines whether this "
        "offline gap survives contact with real behavior."
    )
    return report


if __name__ == "__main__":
    hist = pd.read_csv("data/historical_logs.csv")
    treatment, train_df, test_df = train(hist)
    as_of_day = int(hist["posted_day"].max())
    report = evaluate(test_df, treatment, as_of_day)

    with open("artifacts/offline_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
