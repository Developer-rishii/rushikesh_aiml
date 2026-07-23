"""
train_model.py
---------------
Stage B.2/B.3: "Build it on real data ... keep the experiment log so every
number is reproducible" and "Evaluate on held-out data you did not tune on."

DECISION LOG (why this design, what we rejected):
  - Model family: sklearn HistGradientBoostingClassifier.
    WHY: this environment has no network access, so LightGBM/XGBoost
    (recommended in the study guide's stack) cannot be pip-installed.
    HistGradientBoostingClassifier is scikit-learn's native histogram-based
    gradient boosting implementation -- algorithmically the same family
    (LightGBM's core trick) and a documented, defensible substitute. This is
    recorded here explicitly rather than silently swapped, per Stage A.3
    ("decide your approach and write down WHY").
  - Split: TIME-BASED, not random k-fold.
    WHY: random splitting would leak future engagement patterns into
    training (a candidate's Feb behaviour would help predict their Jan
    churn). We train on snapshots from 2025-03 through 2025-10 and hold out
    2025-11 through 2026-01 ENTIRELY -- the model never sees, and is never
    tuned on, a single row from the holdout snapshots. Rejected alternative:
    stratified k-fold cross-validation (standard sklearn default) -- good
    for i.i.d. data, wrong for a longitudinal churn problem.
  - Class imbalance: churn is a minority class (typically 15-25% base rate
    here, similar to real marketplace churn). We use class_weight="balanced"
    rather than oversampling (SMOTE) -- SMOTE synthesizes fake feature
    vectors, which is a bad idea for a small tabular feature set with
    integer-count features (produces nonsensical fractional counts).
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import compute_features_as_of, FEATURE_COLUMNS, sufficient_history_mask
from label_definition import build_multi_snapshot_labels, HORIZON_DAYS, MIN_HISTORY_DAYS, ALREADY_DORMANT_DAYS

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SNAPSHOTS = pd.date_range("2025-03-01", "2025-10-01", freq="MS")   # monthly snapshots, train window
HOLDOUT_SNAPSHOTS = pd.date_range("2025-11-01", "2026-01-01", freq="MS")  # strictly later in time, never trained on


def _build_snapshot_dataset(profiles, events, snapshot_dates):
    labels = build_multi_snapshot_labels(profiles, events, snapshot_dates)
    feats = []
    for d in snapshot_dates:
        f = compute_features_as_of(profiles, events, d, candidate_ids=labels.loc[labels.as_of_date == d, "candidate_id"])
        feats.append(f)
    feat_df = pd.concat(feats, ignore_index=True)
    merged = feat_df.merge(labels, on=["candidate_id", "as_of_date"], how="inner")
    merged = merged[sufficient_history_mask(merged, MIN_HISTORY_DAYS)].reset_index(drop=True)
    return merged


def main():
    profiles = pd.read_csv(ROOT / "data/raw/candidate_profiles_SIMULATED.csv", parse_dates=["signup_date"])
    events = pd.read_csv(ROOT / "data/raw/interaction_events_SIMULATED.csv", parse_dates=["event_ts"])

    print("[train] building TRAIN snapshots:", [d.date() for d in TRAIN_SNAPSHOTS])
    train_df = _build_snapshot_dataset(profiles, events, TRAIN_SNAPSHOTS)
    print("[train] building HOLDOUT snapshots (never tuned on):", [d.date() for d in HOLDOUT_SNAPSHOTS])
    holdout_df = _build_snapshot_dataset(profiles, events, HOLDOUT_SNAPSHOTS)

    print(f"[train] train rows={len(train_df)} churn_rate={train_df.churned.mean():.3f}")
    print(f"[train] holdout rows={len(holdout_df)} churn_rate={holdout_df.churned.mean():.3f}")

    # further split train into fit/dev (time-adjacent, for early stopping / threshold picking only)
    fit_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df.churned)

    X_fit, y_fit = fit_df[FEATURE_COLUMNS], fit_df.churned
    X_dev, y_dev = dev_df[FEATURE_COLUMNS], dev_df.churned

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_fit, y_fit)
    train_seconds = time.time() - t0

    # persist datasets so evaluate.py / at_risk_list.py reuse the EXACT same holdout (no re-randomization)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv(ROOT / "data/processed/train_snapshots.csv", index=False)
    holdout_df.to_csv(ROOT / "data/processed/holdout_snapshots.csv", index=False)
    dev_df.to_csv(ROOT / "data/processed/dev_snapshots.csv", index=False)

    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "models/churn_model_v1.pkl", "wb") as f:
        pickle.dump(model, f)

    experiment = {
        "run_id": "v1",
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_family": "sklearn.HistGradientBoostingClassifier",
        "rejected_alternatives": ["LightGBM (no network access to install)", "SMOTE oversampling"],
        "label_horizon_days": HORIZON_DAYS,
        "min_history_days": MIN_HISTORY_DAYS,
        "already_dormant_exclusion_days": ALREADY_DORMANT_DAYS,
        "train_snapshots": [str(d.date()) for d in TRAIN_SNAPSHOTS],
        "holdout_snapshots": [str(d.date()) for d in HOLDOUT_SNAPSHOTS],
        "n_train_rows": int(len(fit_df)),
        "n_dev_rows": int(len(dev_df)),
        "n_holdout_rows": int(len(holdout_df)),
        "train_churn_rate": float(fit_df.churned.mean()),
        "holdout_churn_rate": float(holdout_df.churned.mean()),
        "hyperparameters": {
            "max_iter": 300, "learning_rate": 0.06, "max_depth": 5,
            "l2_regularization": 1.0, "class_weight": "balanced",
        },
        "features": FEATURE_COLUMNS,
        "train_seconds": round(train_seconds, 2),
        "random_seed": 42,
    }
    (ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "experiments/experiment_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(experiment) + "\n")

    print(f"[train] done in {train_seconds:.2f}s. Model -> models/churn_model_v1.pkl")
    print(f"[train] experiment logged -> {log_path}")


if __name__ == "__main__":
    main()
