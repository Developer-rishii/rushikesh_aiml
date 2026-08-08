"""
Stage B.2/B.3 — Build on real data, evaluate honestly against a baseline.
Pointwise learning-to-rank (LightGBM) predicting P(shortlisted) as the ranking
score. Trained on TRAIN-time features only (recency_feature_train), which is
the correct pipeline; drift_monitor.py separately proves serving would skew.
"""
import pandas as pd, numpy as np, json, os, joblib, hashlib
from datetime import datetime, timezone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = f"{BASE}/data"
MODELS = f"{BASE}/models"
LOGS = f"{BASE}/logs"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

FEATURES = ["years_experience", "skill_match_score", "profile_completeness",
            "seniority_level", "req_skill_score", "recency_feature_train"]
LABEL = "shortlisted"
GROUP_COL = "job_id"  # ranking is per-job

df = pd.read_csv(f"{DATA}/interactions.csv")

# Split by job_id (group) so we don't leak the same job's candidates across train/test
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df[GROUP_COL]))
train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

X_train, y_train = train_df[FEATURES], train_df[LABEL]
X_test, y_test = test_df[FEATURES], test_df[LABEL]

model_type = "lightgbm_lambdarank" if HAVE_LGB else "logistic_regression_pointwise"

if HAVE_LGB:
    train_df_sorted = train_df.sort_values(GROUP_COL)
    test_df_sorted = test_df.sort_values(GROUP_COL)
    Xtr = train_df_sorted[FEATURES]; ytr = train_df_sorted[LABEL]
    Xte = test_df_sorted[FEATURES]; yte = test_df_sorted[LABEL]
    group_train = train_df_sorted.groupby(GROUP_COL).size().values
    group_test = test_df_sorted.groupby(GROUP_COL).size().values
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=150, learning_rate=0.05,
                            num_leaves=15, min_child_samples=20, random_state=42, verbosity=-1)
    model.fit(Xtr, ytr, group=group_train)
    test_df_sorted = test_df_sorted.copy()
    test_df_sorted["score"] = model.predict(Xte)
    scored = test_df_sorted
else:
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    test_df = test_df.copy()
    test_df["score"] = model.predict_proba(X_test)[:, 1]
    scored = test_df

# --- BASELINE: naive rank by skill_match_score only (what PlaceMux would ship without ML) ---
scored["baseline_score"] = scored["skill_match_score"]

def ndcg_at_k(group, score_col, label_col="shortlisted", k=10):
    g = group.sort_values(score_col, ascending=False).head(k)
    gains = (2 ** g[label_col].values - 1)
    discounts = 1 / np.log2(np.arange(2, len(g) + 2))
    dcg = (gains * discounts).sum()
    ideal = group.sort_values(label_col, ascending=False).head(k)
    igains = (2 ** ideal[label_col].values - 1)
    idcg = (igains * discounts[:len(ideal)]).sum()
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(group, score_col, label_col="shortlisted", k=10):
    g = group.sort_values(score_col, ascending=False).head(k)
    return g[label_col].mean() if len(g) else 0.0

def average_precision(group, score_col, label_col="shortlisted"):
    g = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    hits = g[label_col].values
    if hits.sum() == 0:
        return 0.0
    precisions = [hits[:i+1].mean() for i in range(len(hits)) if hits[i] == 1]
    return float(np.mean(precisions))

def eval_metrics(scored_df, score_col):
    groups = scored_df.groupby(GROUP_COL)
    ndcgs = groups.apply(lambda g: ndcg_at_k(g, score_col)).values
    precs = groups.apply(lambda g: precision_at_k(g, score_col)).values
    maps = groups.apply(lambda g: average_precision(g, score_col)).values
    return {
        "nDCG@10": round(float(np.mean(ndcgs)), 4),
        "precision@10": round(float(np.mean(precs)), 4),
        "MAP": round(float(np.mean(maps)), 4),
        "n_jobs_evaluated": int(len(groups)),
    }

model_metrics = eval_metrics(scored, "score")
baseline_metrics = eval_metrics(scored, "baseline_score")

# --- Simulated online effect (honest gap reporting, per Stage B.3) ---
# We DO NOT claim a real A/B test (none was run). We report expected online CTR/apply-rate
# as a held-out behavioural proxy, and flag explicitly that this is not a live online result.
online_proxy = scored.groupby(GROUP_COL).apply(
    lambda g: g.sort_values("score", ascending=False).head(10)["applied"].mean()
).mean()
online_proxy_baseline = scored.groupby(GROUP_COL).apply(
    lambda g: g.sort_values("baseline_score", ascending=False).head(10)["applied"].mean()
).mean()

results = {
    "model_type": model_type,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "features": FEATURES,
    "offline_model_metrics": model_metrics,
    "offline_baseline_metrics": baseline_metrics,
    "offline_lift_nDCG10_pct": round(
        (model_metrics["nDCG@10"] - baseline_metrics["nDCG@10"]) / max(baseline_metrics["nDCG@10"], 1e-6) * 100, 2),
    "online_proxy_apply_rate_top10_model": round(float(online_proxy), 4),
    "online_proxy_apply_rate_top10_baseline": round(float(online_proxy_baseline), 4),
    "online_proxy_caveat": "No live A/B test was run; this is a held-out behavioural proxy from logged "
                            "data, NOT a claim of validated online lift. Real online validation is an "
                            "explicit dependency handed off in Stage E / Section 13.",
    "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
}

with open(f"{MODELS}/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

joblib.dump(model, f"{MODELS}/ranker.joblib")

# Model registry entry (MLflow-equivalent, disciplined manual registry per Section 6)
model_bytes = open(f"{MODELS}/ranker.joblib", "rb").read()
model_hash = hashlib.sha256(model_bytes).hexdigest()[:16]
data_manifest = json.load(open(f"{DATA}/data_manifest.json"))

registry_entry = {
    "model_version": "v1.0.0",
    "model_hash_sha256_16": model_hash,
    "model_type": model_type,
    "trained_at": results["trained_at"],
    "training_dataset_hash": data_manifest["dataset_sha256_16"],
    "features": FEATURES,
    "excluded_features": ["protected_group (fairness-only, never a model input)"],
}
registry_path = f"{MODELS}/model_registry.json"
registry = json.load(open(registry_path)) if os.path.exists(registry_path) else {"versions": []}
registry["versions"].append(registry_entry)
json.dump(registry, open(registry_path, "w"), indent=2)

# Append to reproducible experiment log (Stage B.2 requirement)
with open(f"{LOGS}/experiment_log.jsonl", "a") as f:
    f.write(json.dumps({"event": "train", **results}) + "\n")

print(json.dumps(results, indent=2))
