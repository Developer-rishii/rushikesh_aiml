"""
train_model.py
---------------
Trains the candidate-ranking model (pointwise learning-to-rank proxy:
GradientBoostingRegressor predicting a graded relevance label built from
click/applied/shortlisted). Evaluates honestly against a held-out set of
JOBS the model never trained on (group split, not a random row split --
a random row split would leak job-level signal and inflate the score).

Reports nDCG@10, MAP@10, precision@10 for:
  (a) the trained model
  (b) a naive baseline (rank candidates by skill_match alone)
so the gap between "model" and "the obvious heuristic" is explicit,
per the Definition-of-Done requirement to evaluate against a baseline.
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data/interaction_logs.csv"
MODEL_PATH = "models/ranker_model.joblib"
REGISTRY_PATH = "models/model_registry.json"
FEATURES = [
    "exp_years", "skill_match", "education_score", "location_match",
    "past_response_rate", "job_seniority", "job_urgency_score",
    "job_num_applicants_so_far",
]
LABEL = "relevance"
K = 10


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(true_rels, scores, k):
    order = np.argsort(-scores)
    ranked_true = np.asarray(true_rels)[order]
    ideal = np.sort(true_rels)[::-1]
    dcg = dcg_at_k(ranked_true, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(true_rels, scores, k):
    order = np.argsort(-scores)[:k]
    hits = (np.asarray(true_rels)[order] > 0).sum()
    return hits / k


def average_precision_at_k(true_rels, scores, k):
    order = np.argsort(-scores)[:k]
    rel = (np.asarray(true_rels)[order] > 0).astype(int)
    if rel.sum() == 0:
        return 0.0
    cum_hits = np.cumsum(rel)
    precisions = cum_hits / (np.arange(len(rel)) + 1)
    return float(np.sum(precisions * rel) / rel.sum())


def evaluate_per_job(df, score_col, k=K):
    ndcgs, maps, precs = [], [], []
    for job_id, g in df.groupby("job_id"):
        y = g[LABEL].values
        s = g[score_col].values
        ndcgs.append(ndcg_at_k(y, s, k))
        maps.append(average_precision_at_k(y, s, k))
        precs.append(precision_at_k(y, s, k))
    return {
        f"nDCG@{k}": float(np.mean(ndcgs)),
        f"MAP@{k}": float(np.mean(maps)),
        f"precision@{k}": float(np.mean(precs)),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Group split BY JOB so evaluation jobs are entirely unseen in training.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    train_idx, test_idx = next(splitter.split(df, groups=df["job_id"]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    assert set(train_df.job_id).isdisjoint(set(test_df.job_id)), "job leakage between train/test!"

    X_train, y_train = train_df[FEATURES], train_df[LABEL]
    X_test = test_df[FEATURES]

    t0 = time.time()
    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=7
    )
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    test_df["model_score"] = model.predict(X_test)
    test_df["baseline_score"] = test_df["skill_match"]  # naive heuristic baseline

    model_metrics = evaluate_per_job(test_df, "model_score")
    baseline_metrics = evaluate_per_job(test_df, "baseline_score")

    gap = {k: round(model_metrics[k] - baseline_metrics[k], 4) for k in model_metrics}

    print("=== Held-out evaluation (jobs never seen in training) ===")
    print("Model   :", model_metrics)
    print("Baseline:", baseline_metrics)
    print("Gap (model - baseline):", gap)
    print(f"Train time: {train_seconds:.2f}s on {len(train_df):,} rows")

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

    registry_entry = {
        "version": "v1",
        "trained_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_train_rows": int(len(train_df)),
        "n_train_jobs": int(train_df.job_id.nunique()),
        "n_test_rows": int(len(test_df)),
        "n_test_jobs": int(test_df.job_id.nunique()),
        "features": FEATURES,
        "model_type": "GradientBoostingRegressor(pointwise LTR proxy)",
        "held_out_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "gap_vs_baseline": gap,
        "train_seconds": round(train_seconds, 3),
        "artifact_path": MODEL_PATH,
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump([registry_entry], f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved model registry (versioned) -> {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
