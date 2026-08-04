"""
train_ranker.py — Stage B: "A pilot run on a real (or realistic) enterprise dataset"

Decision log (guide requires: "write down WHY, including what you rejected"):
  Chosen:   GradientBoostingRegressor (pointwise LTR proxy) trained on the
            tenant's logged impressions, target = weighted engagement
            (click + 3*shortlist + 10*hire), which approximates a graded
            relevance label without needing hand-labeled relevance.
  Rejected: LightGBM/XGBoost LambdaMART -- unavailable in this offline
            sandbox (no network to install). Documented as a known gap:
            production should swap in a listwise LambdaMART objective,
            since pointwise regression optimizes accuracy per-row, not
            the ORDER of results, which the guide explicitly flags as
            the thing that actually drives outcomes.
  Rejected: Fine-tuning a full per-tenant embedding model -- too heavy
            for a pilot dry-run; policy-layer adjustment (feature
            weights) chosen instead so it can be swapped per tenant fast.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")


def load_data():
    cands = pd.read_csv(f"{DATA}/candidates.csv")
    jobs = pd.read_csv(f"{DATA}/jobs.csv")
    impressions = pd.read_csv(f"{DATA}/impressions_log.csv")
    return cands, jobs, impressions


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(EXPER, exist_ok=True)

    cands, jobs, impressions = load_data()
    df = build_features(impressions, cands, jobs)
    df["label"] = df["clicked"] + 3 * df["shortlisted"] + 10 * df["hired"]

    # Held-out split by job_id (group split) so we evaluate on jobs the
    # model did not tune on -- guide: "Evaluate on held-out data you did
    # not tune on."
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["job_id"]))
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    model = GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3, learning_rate=0.08)
    model.fit(train[FEATURE_COLUMNS], train["label"])

    joblib.dump(model, f"{MODELS}/ranker_v1.joblib")

    # Baseline to beat: pure skill_overlap ranking (what the tenant does today, manually)
    baseline_col = "skill_overlap"

    experiment_log = {
        "model_version": "ranker_v1",
        "trained_on_rows": int(len(train)),
        "held_out_rows": int(len(test)),
        "held_out_jobs": sorted(test["job_id"].unique().tolist()),
        "features": FEATURE_COLUMNS,
        "target": "clicked + 3*shortlisted + 10*hired",
        "baseline": baseline_col,
        "chosen_approach": "GradientBoostingRegressor pointwise proxy (see module docstring for rejected alternatives)",
        "reproducible_seed": 42,
    }
    with open(f"{EXPER}/experiment_log.json", "w") as f:
        json.dump(experiment_log, f, indent=2)

    test.to_csv(f"{EXPER}/held_out_test_set.csv", index=False)
    print("Trained ranker_v1 on", len(train), "rows; held out", len(test), "rows across",
          test["job_id"].nunique(), "jobs.")


if __name__ == "__main__":
    main()
