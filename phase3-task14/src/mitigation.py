"""
Stage C - Mitigations applied and re-measured.

Approach chosen: PRE-PROCESSING via Kamiran & Calders reweighing.
Each training row is reweighted so that, in expectation, group and
label are statistically independent going INTO training:

    w(group, label) = ( P(group) * P(label) ) / P(group, label)

Alternative approaches considered and REJECTED (see reports/bias_audit_report.md
for the full write-up):
  - In-processing (fairness-constrained optimization, e.g. exponentiated
    gradient): rejected because it requires re-deriving the training
    objective per model family and is harder to audit independently -
    reweighing keeps the model class fixed, so a reviewer can diff the
    two coefficient vectors directly.
  - Post-processing (per-group threshold adjustment for equalized odds):
    rejected as the PRIMARY fix because setting different decision
    thresholds by protected group is legally risky in Indian hiring
    context (can look like explicit differential treatment even if the
    intent is corrective) - kept as a documented fallback only, not used
    here.

This keeps the same feature contract (features.py) - mitigation changes
HOW we train, not WHAT the model is allowed to see.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

from paths import EXPERIMENTS_DIR
from features import PROTECTED_ATTRIBUTE, LABEL
from train_model import load_split, train, evaluate, build_features
from fairness_metrics import fairness_report


def compute_reweighing_weights(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    weights = np.ones(n)
    groups = df[PROTECTED_ATTRIBUTE].values
    labels = df[LABEL].values

    for g in np.unique(groups):
        for y in np.unique(labels):
            mask = (groups == g) & (labels == y)
            p_g = (groups == g).mean()
            p_y = (labels == y).mean()
            p_gy = mask.mean()
            if p_gy > 0:
                weights[mask] = (p_g * p_y) / p_gy
    return weights


if __name__ == "__main__":
    train_df, test_df = load_split()

    weights = compute_reweighing_weights(train_df)
    print("Sample weight ranges by (gender,label):")
    for g in ["M", "F"]:
        for y in [0, 1]:
            mask = (train_df[PROTECTED_ATTRIBUTE] == g) & (train_df[LABEL] == y)
            if mask.sum():
                print(f"  gender={g} label={y} n={mask.sum():5d} weight={weights[mask][0]:.4f}")

    clf, scaler = train(train_df, sample_weight=weights)
    offline, fairness, proba, y_pred = evaluate(clf, scaler, test_df)

    print("\n=== OFFLINE METRICS (AFTER mitigation, held-out) ===")
    print(json.dumps(offline, indent=2))
    print("=== FAIRNESS AUDIT (AFTER mitigation) ===")
    print(json.dumps(fairness, indent=2))

    joblib.dump({"clf": clf, "scaler": scaler}, os.path.join(EXPERIMENTS_DIR, "model_mitigated.joblib"))
    test_df_out = test_df.copy()
    test_df_out["proba"] = proba
    test_df_out["y_pred"] = y_pred
    test_df_out.to_csv(os.path.join(EXPERIMENTS_DIR, "test_predictions_mitigated.csv"), index=False)

    with open(os.path.join(EXPERIMENTS_DIR, "results_after_mitigation.json"), "w") as f:
        json.dump({"offline_metrics": offline, "fairness_audit": fairness}, f, indent=2)

    # Before/after comparison table for the demo
    with open(os.path.join(EXPERIMENTS_DIR, "results_before_mitigation.json")) as f:
        before = json.load(f)

    comparison = {
        "demographic_parity_diff": {"before": before["fairness_audit"]["demographic_parity_diff"], "after": fairness["demographic_parity_diff"]},
        "equal_opportunity_diff": {"before": before["fairness_audit"]["equal_opportunity_diff"], "after": fairness["equal_opportunity_diff"]},
        "auc": {"before": before["offline_metrics"]["auc"], "after": offline["auc"]},
        "recall@0.5": {"before": before["offline_metrics"]["recall@0.5"], "after": offline["recall@0.5"]},
    }
    with open(os.path.join(EXPERIMENTS_DIR, "before_after_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== BEFORE / AFTER ===")
    print(json.dumps(comparison, indent=2))
