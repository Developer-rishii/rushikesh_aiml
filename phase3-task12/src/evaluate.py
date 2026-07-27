"""
Stage C: "Offline eval (precision@k, coverage, diversity) vs baseline."
Evaluated strictly on the held-out test split written by train.py (data the
model was not tuned on), never on training data.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from recommender import recommend_for_candidate, SKILLS
from baseline import popularity_baseline_topk

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "model_registry")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_log.json")
K = 10


def latest_version():
    with open(LOG_PATH) as f:
        log = json.load(f)
    return log[-1]


def ndcg_at_k(relevant_set, ranked_ids, k=K):
    dcg = 0.0
    for i, jid in enumerate(ranked_ids[:k]):
        if jid in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits)) or 1e-9
    return dcg / idcg


def diversity_at_k(ranked_ids, jobs_idx):
    if len(ranked_ids) < 2:
        return 0.0
    vecs = np.array([jobs_idx.loc[j, SKILLS].values.astype(float) for j in ranked_ids])
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1
    sim = (vecs @ vecs.T) / (norm @ norm.T)
    n = len(ranked_ids)
    off_diag = sim[np.triu_indices(n, k=1)]
    return float(1 - off_diag.mean())  # higher = more diverse


def main():
    entry = latest_version()
    model = joblib.load(entry["model_path"])
    test_split = pd.read_csv([f for f in glob.glob(os.path.join(REG_DIR, "test_split_*.csv"))
                               if entry["version"] in f][0])

    candidates = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    jobs_idx = jobs.set_index("job_id")

    # ground truth relevance per candidate = jobs they actually applied to, in held-out split
    relevant = test_split[test_split["label"] == 1].groupby("candidate_id")["job_id"].apply(set)
    eval_candidates = [c for c in relevant.index if c in candidates["candidate_id"].values]
    eval_candidates = eval_candidates[:150]  # keep eval fast; representative sample

    baseline_topk = popularity_baseline_topk(jobs, k=K)

    model_precisions, model_ndcgs, model_diversities = [], [], []
    base_precisions, base_ndcgs, base_diversities = [], [], []
    model_catalog_hits, base_catalog_hits = set(), set()

    for cid in eval_candidates:
        rel = relevant[cid]

        model_recs = [r["job_id"] for r in recommend_for_candidate(model, cid, candidates, jobs, k=K)]
        model_catalog_hits.update(model_recs)
        model_precisions.append(len(set(model_recs) & rel) / K)
        model_ndcgs.append(ndcg_at_k(rel, model_recs))
        model_diversities.append(diversity_at_k(model_recs, jobs_idx))

        base_catalog_hits.update(baseline_topk)
        base_precisions.append(len(set(baseline_topk) & rel) / K)
        base_ndcgs.append(ndcg_at_k(rel, baseline_topk))
        base_diversities.append(diversity_at_k(baseline_topk, jobs_idx))

    results = {
        "n_eval_candidates": len(eval_candidates),
        "k": K,
        "model": {
            "precision_at_k": float(np.mean(model_precisions)),
            "ndcg_at_k": float(np.mean(model_ndcgs)),
            "diversity_at_k": float(np.mean(model_diversities)),
            "catalog_coverage": len(model_catalog_hits) / len(jobs),
        },
        "baseline_popularity": {
            "precision_at_k": float(np.mean(base_precisions)),
            "ndcg_at_k": float(np.mean(base_ndcgs)),
            "diversity_at_k": float(np.mean(base_diversities)),
            "catalog_coverage": len(base_catalog_hits) / len(jobs),
        },
    }
    results["model_beats_baseline"] = {
        "precision_at_k": results["model"]["precision_at_k"] > results["baseline_popularity"]["precision_at_k"],
        "ndcg_at_k": results["model"]["ndcg_at_k"] > results["baseline_popularity"]["ndcg_at_k"],
        "coverage": results["model"]["catalog_coverage"] > results["baseline_popularity"]["catalog_coverage"],
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments", f"offline_eval_{entry['version']}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
