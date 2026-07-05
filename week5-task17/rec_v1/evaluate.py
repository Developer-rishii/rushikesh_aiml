"""
evaluate.py

Evaluates the trained ML model against the baseline on the held-out test set.
Computes Precision, Recall, FPR, AUC-ROC globally and per-segment.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, precision_recall_curve

from features import FeatureEngineer
from baseline import BaselineRanker

def run_evaluation():
    print("=== Evaluation (Held-out Test Set) ===\\n")
    
    # 1. Load context and test data
    fe = FeatureEngineer()
    fe.load_context()
    
    # Load model and priors
    artifact = joblib.load("model.pkl")
    model = artifact["model"]
    features_to_use = artifact["features_to_use"]
    fe.priors = artifact["college_priors"]
    
    test_outcomes = pd.read_csv("data/test_outcomes.csv")
    print(f"Test Set: {len(test_outcomes)} candidate interactions\\n")
    
    # 2. Extract features
    X_test_full = fe.transform(test_outcomes)
    X_test_sub = X_test_full[features_to_use]
    y_test = test_outcomes["was_hired"].values
    
    # 3. Predict Baseline
    baseline = BaselineRanker()
    y_base_proba = baseline.predict_proba(X_test_full)[:, 1]
    y_base_pred = (y_base_proba > 0.5).astype(int)
    
    # 4. Predict Model
    y_mod_proba = model.predict_proba(X_test_sub)[:, 1]
    y_mod_pred = (y_mod_proba > 0.5).astype(int)
    
    # 5. Compute metrics function
    def compute_metrics(y_true, y_pred, y_proba):
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {"AUC": auc, "Precision": prec, "Recall": rec, "FPR": fpr}
        
    global_base = compute_metrics(y_test, y_base_pred, y_base_proba)
    global_mod = compute_metrics(y_test, y_mod_pred, y_mod_proba)
    
    print("--- Global Metrics ---")
    print(f"{'Metric':<12} | {'Baseline':<10} | {'ML Model':<10} | {'Delta':<10}")
    print("-" * 50)
    for k in ["AUC", "Precision", "Recall", "FPR"]:
        delta = global_mod[k] - global_base[k]
        print(f"{k:<12} | {global_base[k]:.4f}     | {global_mod[k]:.4f}     | {delta:+.4f}")
        
    # 6. Segment Analysis (by College and Seniority)
    print("\\n--- Segment Breakdown (AUC) ---")
    
    test_outcomes["college_id"] = X_test_full["college_id"]
    test_outcomes["seniority"] = X_test_full["seniority_level"]
    test_outcomes["base_proba"] = y_base_proba
    test_outcomes["mod_proba"] = y_mod_proba
    
    print("\\nBy College (Top 3 by volume):")
    colleges = test_outcomes["college_id"].value_counts().head(3).index
    for col in colleges:
        sub = test_outcomes[test_outcomes["college_id"] == col]
        try:
            b_auc = roc_auc_score(sub["was_hired"], sub["base_proba"])
            m_auc = roc_auc_score(sub["was_hired"], sub["mod_proba"])
            print(f"  {col:<10} | Base AUC: {b_auc:.3f} | Mod AUC: {m_auc:.3f}")
        except ValueError:
            pass # Handle colleges with only one class in test split
            
    print("\\nBy Seniority Level:")
    for sen in sorted(test_outcomes["seniority"].unique()):
        sub = test_outcomes[test_outcomes["seniority"] == sen]
        try:
            b_auc = roc_auc_score(sub["was_hired"], sub["base_proba"])
            m_auc = roc_auc_score(sub["was_hired"], sub["mod_proba"])
            print(f"  Level {sen:<4} | Base AUC: {b_auc:.3f} | Mod AUC: {m_auc:.3f}")
        except ValueError:
            pass
            
if __name__ == "__main__":
    run_evaluation()
