"""
Stage B step 2 ("build on real data") + step 3 ("evaluate honestly").
Pointwise learning-to-rank via GradientBoostingClassifier (lightgbm unavailable,
offline sandbox has no network -- substitution logged in experiments/experiment_log.json
as a deliberate, documented decision per the study guide's "write down WHY, including
what you rejected" instruction).

Label = applied (strongest signal in the funnel: impression -> click -> apply).
Negative sampling: for each candidate, sample unseen jobs as negatives so the
model learns more than "popular vs not".
"""
import os, json, hashlib, sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_frame, FEATURE_COLUMNS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "model_registry")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_log.json")


def load_data():
    candidates = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    return candidates, jobs, interactions


def make_training_set(candidates, jobs, interactions, rng):
    pos = interactions[["candidate_id", "job_id", "applied"]].rename(columns={"applied": "label"})
    # negative sampling: unseen (candidate, job) pairs, 1x per positive-ish to keep balance reasonable
    all_jobs = jobs["job_id"].values
    seen = set(zip(interactions["candidate_id"], interactions["job_id"]))
    neg_rows = []
    n_neg = len(interactions)
    cand_ids = interactions["candidate_id"].sample(n_neg, replace=True, random_state=0).values
    for cid in cand_ids:
        for _ in range(3):  # a few tries to avoid an already-seen pair
            jid = rng.choice(all_jobs)
            if (cid, jid) not in seen:
                neg_rows.append({"candidate_id": cid, "job_id": jid, "label": 0})
                break
    neg = pd.DataFrame(neg_rows)
    train_df = pd.concat([
        interactions[["candidate_id", "job_id"]].assign(label=interactions["applied"]),
        neg,
    ], ignore_index=True)
    return train_df


def main():
    os.makedirs(REG_DIR, exist_ok=True)
    rng = np.random.default_rng(7)
    candidates, jobs, interactions = load_data()
    train_df = make_training_set(candidates, jobs, interactions, rng)

    X = build_feature_frame(train_df, candidates, jobs)
    y = train_df["label"].values
    groups = train_df["candidate_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_pairs = train_df.iloc[test_idx].reset_index(drop=True)

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
    )
    model.fit(X_train, y_train)

    # save held-out test set + candidate/job frames so evaluate.py works on the SAME split (reproducible)
    version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = os.path.join(REG_DIR, f"model_{version}.joblib")
    joblib.dump(model, model_path)
    test_pairs.assign(**{c: X_test[c].values for c in FEATURE_COLUMNS}).to_csv(
        os.path.join(REG_DIR, f"test_split_{version}.csv"), index=False
    )

    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    train_auc = float(model.score(X_train, y_train))
    test_auc = float(model.score(X_test, y_test))

    entry = {
        "version": version,
        "model_path": model_path,
        "model_sha256_16": model_hash,
        "algorithm": "GradientBoostingClassifier (lightgbm unavailable in sandbox; documented substitution)",
        "features": FEATURE_COLUMNS,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "train_accuracy": train_auc,
        "test_accuracy": test_auc,
        "rejected_alternatives": [
            "Pure collaborative filtering (matrix factorization only) -- rejected: "
            "800 candidates x 300 jobs is too sparse (20k logged impressions, "
            "~8%% density) for CF alone to generalize to unseen candidates/jobs (cold start).",
            "LightGBM listwise LambdaMART -- preferred in production, but unavailable "
            "offline here; pointwise GBDT substituted and logged as a known gap for hand-off.",
        ],
        "decision_rationale": "Hybrid content-based (skill/city/level overlap features) + "
        "learned pointwise ranker chosen because cold-start candidates/jobs are common in a "
        "marketplace and pure CF fails on those; content features generalize immediately.",
    }
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
