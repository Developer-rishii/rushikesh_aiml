"""
train.py
--------
Loads a real-world dataset (Wisconsin Diagnostic Breast Cancer, 569 real
patient samples, 30 numeric features), builds a reproducible sklearn
Pipeline (StandardScaler + LogisticRegression), evaluates it honestly on a
held-out test split, and serializes the fitted pipeline with joblib so it
can be loaded by the Flask service at inference time.

Run:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    # Persist a copy of the real dataset used, for reproducibility / audit.
    np.savetxt(
        DATA_DIR / "breast_cancer_raw.csv",
        np.column_stack([X, y]),
        delimiter=",",
        header=",".join(feature_names + ["target"]),
        comments="",
    )

    # Honest train/val/test split: fixed seed, stratified, unseen test set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Validate on the held-out validation split (model selection signal).
    val_preds = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    # Final honest evaluation on the never-touched test split.
    test_preds = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "test_precision": round(float(precision_score(y_test, test_preds)), 4),
        "test_recall": round(float(recall_score(y_test, test_preds)), 4),
        "test_f1": round(float(f1_score(y_test, test_preds)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "seed": SEED,
        "trained_at_unix": time.time(),
        "sklearn_pipeline_steps": [s[0] for s in pipeline.steps],
        "feature_order": feature_names,
    }

    # Serialize: model artifact + versioned metadata (rollback traceability).
    model_path = MODELS_DIR / "pipeline_v1.joblib"
    joblib.dump(pipeline, model_path)

    # "latest" pointer for the API to load, without losing older versions.
    latest_path = MODELS_DIR / "pipeline_latest.joblib"
    joblib.dump(pipeline, latest_path)

    with open(MODELS_DIR / "metadata_v1.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / "metadata_latest.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
