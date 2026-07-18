"""
bias_classifier.py
Trains a binary classifier to predict whether a recommendation outcome
was biased (driven by protected attributes) vs skill-driven.

Trained on admin-reviewed fairness_labels.csv.
Persists model to src/models/bias_classifier.pkl.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "src" / "models" / "bias_classifier.pkl"
EXP_LOG     = ROOT / "reports" / "experiment_log.jsonl"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for the bias classifier.
    Input: fairness_labels.csv joined with recommendations.csv
    """
    df = df.copy()
    # Encode region
    region_map = {"urban": 2, "semi_urban": 1, "rural": 0}
    df["region_encoded"] = df["region"].map(region_map).fillna(1)
    # Score gap: difference between skill-only and biased score
    df["score_gap"] = (df["match_score"] - df["match_score_biased"]).round(4)
    # Recommendation flip: did the bias change the recommendation?
    df["rec_flipped"] = (
        df["skill_only_recommended"] != df["production_recommended"]
    ).astype(int)
    feature_cols = [
        "college_tier",       # protected attribute proxy
        "region_encoded",     # protected attribute proxy
        "match_score",        # skill-only signal
        "match_score_biased", # production signal
        "score_gap",          # delta due to potential bias
        "rec_flipped",        # whether recommendation changed
        "skill_only_recommended",
        "production_recommended",
    ]
    return df[feature_cols]


def train(labeled_path: Path = None) -> dict:
    if labeled_path is None:
        labeled_path = ROOT / "data" / "fairness_labels.csv"
    recs_path = ROOT / "data" / "recommendations.csv"

    # ── Load & validate ────────────────────────────────────────────────────────
    labeled = pd.read_csv(labeled_path)
    recs    = pd.read_csv(recs_path)

    required_labeled = {"student_id", "job_id", "is_biased_outcome",
                        "match_score", "match_score_biased",
                        "skill_only_recommended", "production_recommended"}
    missing = required_labeled - set(labeled.columns)
    if missing:
        raise ValueError(f"fairness_labels.csv missing columns: {missing}")

    # fairness_labels.csv already contains college_tier and region
    df = labeled.copy()

    X = build_features(df)
    y = df["is_biased_outcome"].astype(int)

    # ── Split by student_id to prevent leakage ────────────────────────────────
    student_ids = df["student_id"].unique()
    train_ids, test_ids = train_test_split(student_ids, test_size=0.25,
                                            random_state=42)
    val_ids, test_ids   = train_test_split(test_ids,    test_size=0.50,
                                            random_state=42)

    train_mask = df["student_id"].isin(train_ids)
    val_mask   = df["student_id"].isin(val_ids)
    test_mask  = df["student_id"].isin(test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(n_estimators=100, max_depth=6,
                                  class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    # ── Threshold tuning on validation ────────────────────────────────────────
    val_proba = clf.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.75, 0.05):
        preds = (val_proba >= t).astype(int)
        if y_val.sum() == 0:
            continue
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    # ── Evaluate on held-out test ─────────────────────────────────────────────
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_thresh).astype(int)

    prec = precision_score(y_test, test_preds, zero_division=0)
    rec  = recall_score(y_test, test_preds, zero_division=0)
    f1   = f1_score(y_test, test_preds, zero_division=0)
    cm   = confusion_matrix(y_test, test_preds)
    fpr  = cm[0, 1] / cm[0].sum() if cm[0].sum() > 0 else 0.0

    # ── Feature importances ───────────────────────────────────────────────────
    importances = dict(zip(X.columns.tolist(),
                           clf.feature_importances_.round(4).tolist()))

    # ── Save model + metadata ─────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(exist_ok=True)
    artifact = {"model": clf, "threshold": best_thresh,
                "feature_cols": X.columns.tolist()}
    joblib.dump(artifact, MODEL_PATH)

    result = {
        "timestamp":       datetime.utcnow().isoformat(),
        "n_train":         int(y_train.sum()),
        "n_test":          int(len(y_test)),
        "threshold":       round(best_thresh, 2),
        "precision":       round(prec, 4),
        "recall":          round(rec,  4),
        "f1":              round(f1,   4),
        "fpr":             round(fpr,  4),
        "feature_importances": importances,
        "model_path":      str(MODEL_PATH),
    }

    EXP_LOG.parent.mkdir(exist_ok=True)
    with open(EXP_LOG, "a") as fh:
        fh.write(json.dumps(result) + "\n")

    return result


def predict_one(student_id: str, job_id: str,
                recs_df: pd.DataFrame = None) -> dict:
    """Score one (student, job) pair for bias risk."""
    artifact = joblib.load(MODEL_PATH)
    clf, threshold, feature_cols = (artifact["model"],
                                     artifact["threshold"],
                                     artifact["feature_cols"])
    if recs_df is None:
        recs_df = pd.read_csv(ROOT / "data" / "recommendations.csv")

    row = recs_df[(recs_df["student_id"] == student_id) &
                  (recs_df["job_id"] == job_id)]
    if row.empty:
        return {"error": f"No record for student={student_id}, job={job_id}"}

    row = row.iloc[0]
    region_map = {"urban": 2, "semi_urban": 1, "rural": 0}
    features = pd.DataFrame([{
        "college_tier":           row["college_tier"],
        "region_encoded":         region_map.get(row["region"], 1),
        "match_score":            row["match_score"],
        "match_score_biased":     row["match_score_biased"],
        "score_gap":              round(row["match_score"] - row["match_score_biased"], 4),
        "rec_flipped":            int(row["recommended"] != row["production_recommended"]),
        "skill_only_recommended": row["recommended"],
        "production_recommended": row["production_recommended"],
    }])[feature_cols]

    proba = clf.predict_proba(features)[0, 1]
    verdict = "⚠️ BIAS RISK" if proba >= threshold else "✅ SKILL-DRIVEN"

    importances = dict(zip(feature_cols, clf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:3]

    reason_parts = [
        f"college_tier={int(row['college_tier'])}",
        f"region={row['region']}",
        f"score_gap={round(row['match_score'] - row['match_score_biased'], 4)}",
    ]
    reason = (
        f"{verdict} — bias_risk_score={proba:.3f} (threshold={threshold}). "
        f"Top features: {', '.join(f'{k}({v:.3f})' for k,v in top_features)}. "
        f"Context: {'; '.join(reason_parts)}."
    )
    return {
        "student_id":          student_id,
        "job_id":              job_id,
        "bias_risk_score":     round(float(proba), 4),
        "verdict":             verdict,
        "threshold_used":      threshold,
        "skill_rec":           int(row["recommended"]),
        "production_rec":      int(row["production_recommended"]),
        "reason":              reason,
    }


if __name__ == "__main__":
    print("Training bias classifier...")
    result = train()
    print(f"\nModel trained and saved to {result['model_path']}")
    print(f"  Threshold:  {result['threshold']}")
    print(f"  Precision:  {result['precision']}")
    print(f"  Recall:     {result['recall']}")
    print(f"  F1:         {result['f1']}")
    print(f"  FPR:        {result['fpr']}")
    print(f"\nTop features:")
    for k, v in sorted(result["feature_importances"].items(),
                        key=lambda x: -x[1])[:4]:
        print(f"    {k}: {v}")
