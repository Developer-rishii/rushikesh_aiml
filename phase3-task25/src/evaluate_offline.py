"""
Stage B.3 - "Evaluate honestly against a baseline"
Baseline = current production system (popularity ranker: rank jobs by
overall historic click rate, no personalization). This is the real bar
PlaceMux must clear per the study guide ("measurably better than a
baseline"). Metrics: nDCG@10, MAP@10, Precision@10, computed per candidate
(query) then averaged - the standard learning-to-rank evaluation protocol
(Sec 5, "Learning-to-rank").
"""
import os, pickle, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from common import get_X

ROOT = os.path.dirname(os.path.dirname(__file__))
EXP_LOG = f"{ROOT}/reports/experiment_log.csv"

def precision_at_k(rel_sorted, k=10):
    top = rel_sorted[:k]
    return float((top >= 2).sum()) / max(len(top), 1)  # relevance>=2 = "applied" counts as relevant

def average_precision_at_k(rel_sorted, k=10):
    top = rel_sorted[:k]
    hits, ap = 0, 0.0
    for i, r in enumerate(top, 1):
        if r >= 2:
            hits += 1
            ap += hits / i
    return ap / max(hits, 1) if hits else 0.0

def eval_ranking(df, score_col, k=10):
    ndcgs, precs, maps = [], [], []
    for qid, g in df.groupby("query_id"):
        if len(g) < 2 or g.relevance.sum() == 0:
            continue
        order = g[score_col].values.argsort()[::-1]
        rel_sorted = g.relevance.values[order]
        true_rel = g.relevance.values.reshape(1, -1)
        pred_score = g[score_col].values.reshape(1, -1)
        try:
            ndcgs.append(ndcg_score(true_rel, pred_score, k=k))
        except Exception:
            continue
        precs.append(precision_at_k(rel_sorted, k))
        maps.append(average_precision_at_k(rel_sorted, k))
    return dict(nDCG_at_10=np.mean(ndcgs), MAP_at_10=np.mean(maps),
                Precision_at_10=np.mean(precs), n_queries=len(ndcgs))

def main():
    df = pd.read_csv(f"{ROOT}/data/logs.csv")
    test = df[df.day >= 20].copy()

    with open(f"{ROOT}/registry/models/ranker_v2.0.pkl", "rb") as f:
        model = pickle.load(f)
    test["model_score"] = model.predict(get_X(test))

    # baseline: production popularity ranker = per-job historic click rate
    pop = df[df.day < 20].groupby("job_id").clicked.mean().rename("baseline_score")
    test = test.merge(pop, on="job_id", how="left")
    test["baseline_score"] = test["baseline_score"].fillna(df[df.day < 20].clicked.mean())

    model_metrics = eval_ranking(test, "model_score")
    baseline_metrics = eval_ranking(test, "baseline_score")

    lift = {k: (model_metrics[k] - baseline_metrics[k]) for k in ["nDCG_at_10", "MAP_at_10", "Precision_at_10"]}
    result = {"model_v2.0": model_metrics, "baseline_popularity": baseline_metrics,
              "absolute_lift": lift, "evaluated_on": "held-out day>=20, untuned", "k": 10}

    os.makedirs(f"{ROOT}/reports", exist_ok=True)
    json.dump(result, open(f"{ROOT}/reports/offline_eval.json", "w"), indent=2)

    with open(EXP_LOG, "a") as f:
        t = int(time.time())
        for k, v in model_metrics.items():
            f.write(f"offline_eval,model,{k},{v},{t}\n")
        for k, v in baseline_metrics.items():
            f.write(f"offline_eval,baseline,{k},{v},{t}\n")

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
