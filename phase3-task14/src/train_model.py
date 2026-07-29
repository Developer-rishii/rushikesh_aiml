"""
Stage B.2 / B.3 - Build on real data + evaluate honestly against a baseline.

We use Logistic Regression on purpose (not a black-box GBM) because
Stage D needs EXACT, faithful per-decision explanations exposed via
an API - a linear model's coefficients ARE the true attribution,
with no approximation error (no SHAP-on-a-blackbox surrogate risk).
This is a deliberate, documented trade-off - see reports/bias_audit_report.md
section "Alternative approaches".

Train/serve skew guard: `build_features()` is the ONLY place feature
computation happens, and both training and the API import it - so
the features seen at serve time cannot silently diverge from training.
"""
import json
import os
import numpy as np
import pandas as pd
from paths import EXPERIMENTS_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
import joblib

from features import MODEL_FEATURES, LABEL, PROTECTED_ATTRIBUTE
from fairness_metrics import fairness_report


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for feature computation (train == serve)."""
    return df[MODEL_FEATURES].copy()


def load_split(path=None, seed=42):
    if path is None:
        from paths import DATA_DIR
        path = os.path.join(DATA_DIR, "interactions_log.csv")
    df = pd.read_csv(path)
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=seed, stratify=df[LABEL]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train(train_df, sample_weight=None):
    X = build_features(train_df)
    y = train_df[LABEL].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y, sample_weight=sample_weight)
    return clf, scaler


def evaluate(clf, scaler, test_df, threshold=0.5):
    X = build_features(test_df)
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    y_true = test_df[LABEL].values

    offline = {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "precision@0.5": round(float(precision_score(y_true, y_pred)), 4),
        "recall@0.5": round(float(recall_score(y_true, y_pred)), 4),
        "positive_rate": round(float(y_pred.mean()), 4),
        "baseline_positive_rate_majority_class": round(float(1 - y_true.mean()), 4),
    }
    fairness = fairness_report(y_true, y_pred, test_df[PROTECTED_ATTRIBUTE].values)
    return offline, fairness, proba, y_pred


if __name__ == "__main__":
    train_df, test_df = load_split()
    clf, scaler = train(train_df)

    offline, fairness, proba, y_pred = evaluate(clf, scaler, test_df)

    print("=== OFFLINE METRICS (held-out, not tuned on) ===")
    print(json.dumps(offline, indent=2))
    print("=== FAIRNESS AUDIT (BEFORE mitigation) ===")
    print(json.dumps(fairness, indent=2))

    joblib.dump({"clf": clf, "scaler": scaler}, os.path.join(EXPERIMENTS_DIR, "model_baseline.joblib"))
    test_df_out = test_df.copy()
    test_df_out["proba"] = proba
    test_df_out["y_pred"] = y_pred
    test_df_out.to_csv(os.path.join(EXPERIMENTS_DIR, "test_predictions_baseline.csv"), index=False)

    with open(os.path.join(EXPERIMENTS_DIR, "results_before_mitigation.json"), "w") as f:
        json.dump({"offline_metrics": offline, "fairness_audit": fairness}, f, indent=2)

    print("\nSaved model + predictions + results_before_mitigation.json")
