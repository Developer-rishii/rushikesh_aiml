"""
evaluate.py — Stage B/C step 3: "Evaluate honestly against a baseline"

Computes nDCG@10, MAP@10, Precision@10 per job on the held-out test set
(jobs unseen at tune time), comparing:
  - baseline: rank by skill_overlap alone (what the tenant does manually today)
  - model:    rank by ranker_v1 predicted score

Also reports the offline-vs-online gap warning required by the guide:
"report the gap between offline metric and expected online effect" --
we do this by simulating a held-out online proxy (actual hire outcomes)
and checking whether the offline ranking gain still predicts hire lift.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) / discounts))


def ndcg_at_k(rels_sorted_by_score, k):
    ideal = sorted(rels_sorted_by_score, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels_sorted_by_score, k) / idcg


def ap_at_k(rels_sorted_by_score, k):
    rels = np.asarray(rels_sorted_by_score)[:k]
    hits, sum_prec = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r > 0:
            hits += 1
            sum_prec += hits / i
    total_rel = np.sum(np.asarray(rels_sorted_by_score) > 0)
    return sum_prec / total_rel if total_rel > 0 else 0.0


def precision_at_k(rels_sorted_by_score, k):
    rels = np.asarray(rels_sorted_by_score)[:k]
    return float(np.mean(rels > 0)) if len(rels) else 0.0


def eval_ranking(df, score_col, rel_col, k=10):
    ndcgs, maps, precs = [], [], []
    for job_id, g in df.groupby("job_id"):
        g_sorted = g.sort_values(score_col, ascending=False)
        rels = g_sorted[rel_col].tolist()
        ndcgs.append(ndcg_at_k(rels, k))
        maps.append(ap_at_k(rels, k))
        precs.append(precision_at_k(rels, k))
    return {
        "nDCG@10": float(np.mean(ndcgs)),
        "MAP@10": float(np.mean(maps)),
        "Precision@10": float(np.mean(precs)),
        "n_jobs_evaluated": len(ndcgs),
    }


def main():
    test = pd.read_csv(f"{EXPER}/held_out_test_set.csv")
    model = joblib.load(f"{MODELS}/ranker_v1.joblib")
    test["model_score"] = model.predict(test[FEATURE_COLUMNS])

    # relevance label for ranking metrics: graded by outcome strength
    test["relevance"] = test["clicked"] + 3 * test["shortlisted"] + 10 * test["hired"]

    baseline_metrics = eval_ranking(test, "skill_overlap", "relevance")
    model_metrics = eval_ranking(test, "model_score", "relevance")

    # Offline-vs-online proxy: does top-10-by-model actually contain more real hires
    # than top-10-by-baseline? This is the "expected online effect" proxy required
    # by the guide -- offline nDCG can go up while real hire capture doesn't.
    def hire_capture_at_k(score_col, k=10):
        caps = []
        for job_id, g in test.groupby("job_id"):
            top = g.sort_values(score_col, ascending=False).head(k)
            caps.append(top["hired"].sum())
        return float(np.mean(caps))

    baseline_hire_capture = hire_capture_at_k("skill_overlap")
    model_hire_capture = hire_capture_at_k("model_score")

    report = {
        "baseline_offline_metrics": baseline_metrics,
        "model_offline_metrics": model_metrics,
        "offline_gain": {
            "nDCG@10_delta": round(model_metrics["nDCG@10"] - baseline_metrics["nDCG@10"], 4),
            "MAP@10_delta": round(model_metrics["MAP@10"] - baseline_metrics["MAP@10"], 4),
            "Precision@10_delta": round(model_metrics["Precision@10"] - baseline_metrics["Precision@10"], 4),
        },
        "online_proxy_hire_capture_at_10": {
            "baseline": baseline_hire_capture,
            "model": model_hire_capture,
            "delta": round(model_hire_capture - baseline_hire_capture, 4),
        },
        "offline_vs_online_gap_note": (
            "Offline nDCG/MAP gains are reported alongside a real-outcome hire-capture "
            "proxy on the SAME held-out jobs. If the hire-capture delta is smaller or "
            "flips sign relative to the nDCG delta, the offline win should NOT be shipped "
            "without an online A/B test -- this is the exact failure mode the guide warns "
            "about ('nDCG going up offline means nothing if applications go down online')."
        ),
    }
    with open(f"{EXPER}/metrics_offline.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
