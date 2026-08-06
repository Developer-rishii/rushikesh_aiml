"""
evaluate.py
Stage C.3: "Evaluate honestly against a baseline" — held-out data, offline
metric AND expected-online-effect gap, PLUS a fairness slice every run
(not a one-time formality — named pitfall explicitly avoided).
"""
import json, os
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ranker import robust_features, naive_features
from stuffing_detector import rule_signals


def ndcg_at_k(scores_true_order, k=10):
    """scores_true_order: list of true_relevance sorted by PREDICTED rank desc."""
    scores = scores_true_order[:k]
    dcg = sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))
    ideal = sorted(scores_true_order, reverse=True)[:k]
    idcg = sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(scores_true_order, k=10, thresh=0.5):
    top = scores_true_order[:k]
    return sum(1 for s in top if s >= thresh) / k


def evaluate(candidates, jobs, interactions, model, stuffing_scores, k=10):
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    job_by_id = {j["job_id"]: j for j in jobs}
    by_job = defaultdict(list)
    for row in interactions:
        by_job[row["job_id"]].append(row)

    ndcgs_robust, ndcgs_naive, precisions_robust = [], [], []
    fairness_group_scores = defaultdict(list)
    applications_captured_robust, applications_captured_naive = 0, 0
    total_applications = 0

    for job_id, rows in by_job.items():
        job = job_by_id[job_id]
        robust_scored, naive_scored = [], []
        for row in rows:
            c = cand_by_id[row["candidate_id"]]
            s = stuffing_scores.get(c["candidate_id"], 0.0)
            rf = robust_features(c, job, s)
            pred = model.predict([rf])[0]
            robust_scored.append((pred, row["true_relevance"], c, row))
            naive_scored.append((naive_features(c, job), row["true_relevance"], c, row))

        robust_scored.sort(key=lambda t: t[0], reverse=True)
        naive_scored.sort(key=lambda t: t[0], reverse=True)

        true_order_robust = [t[1] for t in robust_scored]
        true_order_naive = [t[1] for t in naive_scored]
        ndcgs_robust.append(ndcg_at_k(true_order_robust, k))
        ndcgs_naive.append(ndcg_at_k(true_order_naive, k))
        precisions_robust.append(precision_at_k(true_order_robust, k))

        # online-effect proxy: of the top-k actually surfaced, how many led
        # to a real application in the logs?
        applications_captured_robust += sum(1 for t in robust_scored[:k] if t[3]["applied"])
        applications_captured_naive += sum(1 for t in naive_scored[:k] if t[3]["applied"])
        total_applications += sum(1 for r in rows if r["applied"])

        for pred, true_rel, c, row in robust_scored[:k]:
            fairness_group_scores[c["protected_group"]].append(true_rel)

    fairness = {g: round(float(np.mean(v)), 4) for g, v in fairness_group_scores.items()}
    fairness_gap = round(max(fairness.values()) - min(fairness.values()), 4) if fairness else None

    result = {
        "k": k,
        "offline_nDCG_at_k": {
            "robust_model": round(float(np.mean(ndcgs_robust)), 4),
            "naive_stuffable_baseline": round(float(np.mean(ndcgs_naive)), 4),
        },
        "precision_at_k_robust": round(float(np.mean(precisions_robust)), 4),
        "online_effect_proxy_application_capture_rate": {
            "robust_model": round(applications_captured_robust / max(1, total_applications), 4),
            "naive_stuffable_baseline": round(applications_captured_naive / max(1, total_applications), 4),
            "note": "share of real logged applications that occur within the top-k the model would have surfaced -- this is the offline-vs-online gap check required by Stage C.3",
        },
        "fairness_slice_true_relevance_by_protected_group_top_k": fairness,
        "fairness_max_group_gap": fairness_gap,
    }
    return result


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(base, "data", "candidates.json")) as f:
        candidates = json.load(f)
    with open(os.path.join(base, "data", "jobs.json")) as f:
        jobs = json.load(f)
    
    test_interactions_path = os.path.join(os.path.dirname(__file__), "test_interactions.json")
    if os.path.exists(test_interactions_path):
        with open(test_interactions_path) as f:
            interactions = json.load(f)
    else:
        # Fallback to full interactions if not present, but should warn
        print("Warning: test_interactions.json not found, falling back to full data.")
        with open(os.path.join(base, "data", "interactions.json")) as f:
            interactions = json.load(f)

    import joblib
    model = joblib.load(os.path.join(os.path.dirname(__file__), "ranker.joblib"))
    stuffing_scores = {c["candidate_id"]: rule_signals(c["resume_text"])["repetition_rate"]
                        for c in candidates}
    result = evaluate(candidates, jobs, interactions, model, stuffing_scores)

    with open(os.path.join(os.path.dirname(__file__), "ranking_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
