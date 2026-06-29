import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, confusion_matrix

FEATURES = [
    'tab_switch_count', 'face_count_anomalies', 
    'copy_paste_events', 'time_per_question_zscore', 
    'network_latency_flag', 'webcam_dropout_seconds'
]

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr
    }

def train_and_evaluate(df: pd.DataFrame, model_dir="d:/Placemux-aiml/wek4-task11/src/models"):
    # Ensure un-reviewed rows are never used as ground truth
    reviewed_df = df[df['ground_truth_reviewed'] == 1].copy()
    
    X = reviewed_df[FEATURES]
    y = reviewed_df['confirmed_violation']
    v0_preds = reviewed_df['flagged_by_v0_proctor']

    # Train/Test split
    X_train, X_test, y_train, y_test, v0_train, v0_test = train_test_split(
        X, y, v0_preds, test_size=0.3, random_state=42, stratify=y
    )

    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train_imputed, y_train)

    # Evaluate Model
    y_pred = clf.predict(X_test_imputed)
    model_metrics = calculate_metrics(y_test, y_pred)
    
    # Evaluate Baseline (v0)
    baseline_metrics = calculate_metrics(y_test, v0_test)

    # Save model and imputer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(imputer, os.path.join(model_dir, 'imputer.joblib'))

    # Log experiment
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "features": FEATURES,
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_))
    }
    with open(os.path.join(model_dir, 'experiment_log.json'), 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    return log_entry, clf, imputer

def load_model_and_imputer(model_dir="d:/Placemux-aiml/wek4-task11/src/models"):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    imputer = joblib.load(os.path.join(model_dir, 'imputer.joblib'))
    return clf, imputer

def generate_explanation(row: pd.Series, model, proba: float) -> str:
    from src.data_loader import is_sensor_fault
    
    if is_sensor_fault(row):
        return "Not scored: Sensor fault detected (all signals missing or zero)."

    importances = dict(zip(FEATURES, model.feature_importances_))
    # Sort features by importance
    top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:2]
    
    reasons = []
    for feat, imp in top_features:
        val = row.get(feat)
        if pd.notna(val) and val > 0:
            reasons.append(f"{feat} is {val:.2f}")

    reason_str = "; ".join(reasons) if reasons else "No strong signals."
    
    verdict = "Flagged" if proba > 0.5 else "Clean"
    
    return f"{verdict}: Model confidence {proba:.2f}. Driven mainly by: {reason_str}."

def predict_session(row: pd.Series, model, imputer) -> dict:
    from src.data_loader import is_sensor_fault
    
    v0_flag = int(row['flagged_by_v0_proctor'])
    
    if is_sensor_fault(row):
        return {
            "v0_flag": v0_flag,
            "model_score": None,
            "confidence": None,
            "explanation": "Not scored: Sensor fault detected (all signals missing or zero).",
            "verdict": "no_data"
        }
        
    X = pd.DataFrame([row[FEATURES]])
    X_imputed = imputer.transform(X)
    
    proba = float(model.predict_proba(X_imputed)[0][1])
    verdict = "flagged" if proba > 0.5 else "clean"
    
    # Borderline row confidence handling is inherent to predict_proba 
    # but we will test it later.
    
    explanation = generate_explanation(row, model, proba)
    
    return {
        "v0_flag": v0_flag,
        "model_score": proba,
        "confidence": proba if proba > 0.5 else 1 - proba,
        "explanation": explanation,
        "verdict": verdict
    }
