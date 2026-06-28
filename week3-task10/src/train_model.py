"""
PlaceMux Quality Sign-Off - Model Training
============================================
Trains a RandomForestClassifier on (student, job) features to predict is_good_match.
Persists the trained model to disk, logs experiment metadata.
"""

import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

from src.features import build_feature_matrix, FEATURE_COLS
from src.labeling import label_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_and_save(students: pd.DataFrame, jobs: pd.DataFrame,
                   events: pd.DataFrame) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Build features & labels ---
    feat_df = build_feature_matrix(students, jobs, events)
    labels = label_dataset(students, jobs, events)
    feat_df = feat_df.reset_index(drop=True)
    labels = labels.iloc[: len(feat_df)].reset_index(drop=True)
    feat_df["label"] = labels

    X = feat_df[FEATURE_COLS].values
    y = feat_df["label"].values

    # --- 60 / 20 / 20 split ---
    X_train_val, X_test, y_train_val, y_test, idx_tv, idx_test = train_test_split(
        X, y, feat_df.index, test_size=0.20, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val,
    )

    # --- Train ---
    model_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "random_state": 42,
        "class_weight": "balanced",
    }
    clf = RandomForestClassifier(**model_params)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    # --- Evaluate ---
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    # --- Persist model ---
    model_path = os.path.join(MODEL_DIR, "match_model.joblib")
    joblib.dump(clf, model_path)

    # --- Save test indices for evaluation ---
    test_meta = feat_df.iloc[idx_test][["student_id", "job_id", "application_id"]].copy()
    test_meta["label"] = y_test
    test_meta.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)
    feat_df.to_csv(os.path.join(DATA_DIR, "features_with_labels.csv"), index=False)

    # --- Experiment log ---
    experiment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "RandomForestClassifier",
        "params": model_params,
        "dataset_size": len(feat_df),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "label_distribution": {
            "train_pos": int(y_train.sum()),
            "train_neg": int(len(y_train) - y_train.sum()),
            "test_pos": int(y_test.sum()),
            "test_neg": int(len(y_test) - y_test.sum()),
        },
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "val_precision": round(val_report.get("1", {}).get("precision", 0), 4),
        "val_recall": round(val_report.get("1", {}).get("recall", 0), 4),
        "val_f1": round(val_report.get("1", {}).get("f1-score", 0), 4),
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_report.get("1", {}).get("precision", 0), 4),
        "test_recall": round(test_report.get("1", {}).get("recall", 0), 4),
        "test_f1": round(test_report.get("1", {}).get("f1-score", 0), 4),
        "feature_importances": dict(zip(FEATURE_COLS, [round(f, 4) for f in clf.feature_importances_])),
        "train_time_sec": train_time,
        "model_artifact": model_path,
    }

    log_path = os.path.join(MODEL_DIR, "experiment_log.json")
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
    logs.append(experiment)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[OK] Model trained in {train_time}s -> {model_path}")
    print(f"  |  Train accuracy: {train_acc:.4f}")
    print(f"  |  Val accuracy:   {val_acc:.4f}  (P={val_report.get('1',{}).get('precision',0):.3f}  R={val_report.get('1',{}).get('recall',0):.3f})")
    print(f"  |  Test accuracy:  {test_acc:.4f}  (P={test_report.get('1',{}).get('precision',0):.3f}  R={test_report.get('1',{}).get('recall',0):.3f})")
    print(f"  `  Experiment log: src/models/experiment_log.json")

    return experiment

def load_model():
    model_path = os.path.join(MODEL_DIR, "match_model.joblib")
    return joblib.load(model_path)

def main():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "monetization_events.csv"))
    train_and_save(students, jobs, events)

if __name__ == "__main__":
    main()
