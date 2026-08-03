"""
Stage B — base matching model. Pointwise learning-to-rank (GradientBoosting
regressor predicting relevance/label) — chosen over pairwise/listwise for
this task (see README "Alternatives" for why: with 3 tenants and a per-
tenant *policy layer* on top, a simple, fast, well-calibrated pointwise
score is easier to combine with business rules than a pairwise model whose
raw scores aren't independently meaningful).

We train ONE base model (tenant-agnostic signal quality) and apply
tenant-specific weighting/rules in policy.py — this is deliberate
(see README "Rules on top of the model vs retraining per tenant").
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
import joblib

from features import compute_features, assert_no_protected_attrs, FEATURE_COLUMNS

from config import DATA_DIR

def load_logs(path=DATA_DIR / "logs.pkl"):
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. Please run data_gen.py first.")
    return pd.read_pickle(path)


def train_test_split_by_job(logs, test_size=0.2, seed=42):
    """Split by job_id (group), not by row — prevents leakage of the same
    job's candidates across train/test, which would silently inflate
    offline metrics."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(logs, groups=logs["job_id"]))
    return logs.iloc[train_idx].reset_index(drop=True), logs.iloc[test_idx].reset_index(drop=True)


def train_model(train_df):
    feats = compute_features(train_df)
    assert_no_protected_attrs(FEATURE_COLUMNS)
    X = feats[FEATURE_COLUMNS]
    y = feats["label"]
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    except ImportError:
        print("WARNING: lightgbm not installed. Falling back to sklearn GradientBoostingRegressor.")
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    return model


def score(model, df):
    feats = compute_features(df)
    return model.predict(feats[FEATURE_COLUMNS])


# ---- offline ranking metrics -------------------------------------------
def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    return np.sum(rels / np.log2(np.arange(2, len(rels) + 2)))


def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)
    ideal_order = np.argsort(-np.asarray(y_true))
    dcg = dcg_at_k(np.asarray(y_true)[order], k)
    idcg = dcg_at_k(np.asarray(y_true)[ideal_order], k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true_binary, y_score, k=10):
    order = np.argsort(-y_score)[:k]
    return np.mean(np.asarray(y_true_binary)[order]) if len(order) else 0.0


def average_precision(y_true_binary, y_score):
    order = np.argsort(-y_score)
    y = np.asarray(y_true_binary)[order]
    if y.sum() == 0:
        return 0.0
    hits, precisions = 0, []
    for i, rel in enumerate(y, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return np.mean(precisions)


def evaluate_per_job(df, y_score_col="score", label_col="shortlisted", k=10):
    ndcgs, p_at_k, aps = [], [], []
    for job_id, g in df.groupby("job_id"):
        ndcgs.append(ndcg_at_k(g["label"], g[y_score_col], k))
        p_at_k.append(precision_at_k(g[label_col], g[y_score_col], k))
        aps.append(average_precision(g[label_col], g[y_score_col]))
    return {
        "nDCG@10": float(np.mean(ndcgs)),
        "precision@10": float(np.mean(p_at_k)),
        "MAP": float(np.mean(aps)),
        "n_jobs_evaluated": len(ndcgs),
    }


if __name__ == "__main__":
    logs = load_logs()
    train_df, test_df = train_test_split_by_job(logs)
    model = train_model(train_df)
    joblib.dump(model, DATA_DIR / "model.joblib")

    test_df = test_df.copy()
    test_df["score"] = score(model, test_df)

    # Baseline = skill_overlap only (what PlaceMux likely shipped pre-ML).
    test_df["baseline_score"] = test_df["skill_overlap"] + np.random.normal(0, 1e-6, len(test_df))

    model_metrics = evaluate_per_job(test_df, "score")
    baseline_metrics = evaluate_per_job(test_df, "baseline_score")

    print("=== HELD-OUT EVAL (jobs never seen in training) ===")
    print("Model   :", model_metrics)
    print("Baseline:", baseline_metrics)
    lift = (model_metrics["nDCG@10"] - baseline_metrics["nDCG@10"]) / baseline_metrics["nDCG@10"] * 100
    print(f"nDCG@10 lift over baseline: {lift:.1f}%")

    test_df.to_pickle(DATA_DIR / "test_scored.pkl")
