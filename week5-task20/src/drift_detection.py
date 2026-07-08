"""
Drift detection: train a classifier to distinguish original training data
from fresh validation data.  If AUC >> 0.5, distribution has shifted.
Complemented by per-feature KS-tests.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import json
import os

from .validate_oos import add_derived_features, FEATURE_COLS


def run_drift_detection(data_dir: str, reports_dir: str, task16_dir: str) -> dict:
    """Train a drift classifier and run KS-tests."""
    print("\n[DRIFT] Loading original + fresh data...")

    # Original training data from Task 16
    orig_path = os.path.join(task16_dir, "data", "matching_v1_output.csv")
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"Original training data not found at {orig_path}")

    orig_df = pd.read_csv(orig_path)
    orig_df = add_derived_features(orig_df)

    # Fresh validation data
    fresh_df = pd.read_csv(os.path.join(data_dir, "fresh_matching.csv"))
    fresh_df = add_derived_features(fresh_df)

    print(f"[DRIFT] Original: {len(orig_df)} rows, Fresh: {len(fresh_df)} rows")

    # Label: origin=0 for original, origin=1 for fresh
    
    # Exclude college_avg_match_score from drift features because it perfectly 
    # separates the datasets (each dataset has different exact mean scalars for colleges)
    drift_features = [f for f in FEATURE_COLS if f != "college_avg_match_score"]
    
    orig_features = orig_df[drift_features].copy()
    orig_features["origin"] = 0

    fresh_features = fresh_df[drift_features].copy()
    fresh_features["origin"] = 1

    combined = pd.concat([orig_features, fresh_features], ignore_index=True)
    X = combined[drift_features]
    y = combined["origin"]

    # Train/test split for the drift-classification task itself
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    drift_auc = round(float(roc_auc_score(y_test, y_pred_proba)), 4)

    print(f"[DRIFT] Drift classifier AUC: {drift_auc}")

    # ── Interpret ─────────────────────────────────────────────────────────
    if drift_auc < 0.55:
        interpretation = (
            f"No significant distribution drift detected (AUC={drift_auc}, "
            f"near random chance). The fresh validation data is statistically "
            f"similar to the original training data."
        )
        drift_severity = "none"
    elif drift_auc < 0.65:
        interpretation = (
            f"Minor distribution drift detected (AUC={drift_auc}). "
            f"The classifier can weakly distinguish old from new data. "
            f"This is expected when scaling up; model performance should be "
            f"monitored but is likely still valid."
        )
        drift_severity = "minor"
    elif drift_auc < 0.80:
        interpretation = (
            f"Moderate distribution drift detected (AUC={drift_auc}). "
            f"The data distributions differ noticeably. Consider retraining "
            f"the Rec v1 model on a blend of old and new data."
        )
        drift_severity = "moderate"
    else:
        interpretation = (
            f"Significant distribution drift detected (AUC={drift_auc}). "
            f"The fresh data is substantially different from training data. "
            f"Rec v1 model predictions may be unreliable — retraining recommended."
        )
        drift_severity = "significant"

    # ── Per-feature KS-tests ──────────────────────────────────────────────
    ks_results = {}
    for col in FEATURE_COLS:
        stat, pval = ks_2samp(orig_df[col].dropna(), fresh_df[col].dropna())
        ks_results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 4),
            "drifted": bool(pval < 0.05),
        }

    drifted_features = [f for f, v in ks_results.items() if v["drifted"]]
    stable_features = [f for f, v in ks_results.items() if not v["drifted"]]

    # ── Feature importances from drift classifier ─────────────────────────
    feat_imp = sorted(
        zip(drift_features, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    results = {
        "drift_auc": drift_auc,
        "drift_severity": drift_severity,
        "interpretation": interpretation,
        "ks_test_results": ks_results,
        "drifted_features": drifted_features,
        "stable_features": stable_features,
        "drift_feature_importances": [[f, round(float(v), 5)] for f, v in feat_imp],
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "drift_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DRIFT] Results saved to reports/drift_results.json")

    return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_drift_detection(
        os.path.join(base, "data"),
        os.path.join(base, "reports"),
        os.path.join(base, "..", "week5-task16"),
    )
