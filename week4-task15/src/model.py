import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import joblib
import os

def engineer_features(df):
    # Create derived features
    df = df.copy()
    
    # 1. Signal combination score
    signals = ["tab_switch_count", "face_count_anomalies", "copy_paste_events"]
    # Handle NaN for sensor faults
    df["signal_combo_score"] = df[signals].fillna(0).sum(axis=1)
    
    # 2. Network issue flag derived (high latency AND webcam dropout)
    df["network_issue_derived"] = ((df["network_latency_flag"] == 1) & (df["webcam_dropout_seconds"] > 5)).astype(int)
    
    # Keep original signals as well
    feature_cols = [
        "tab_switch_count", "face_count_anomalies", "copy_paste_events",
        "time_per_question_zscore", "network_latency_flag", "webcam_dropout_seconds",
        "signal_combo_score", "network_issue_derived"
    ]
    
    return df, feature_cols

def train_and_evaluate(df, output_model_path="d:/Placemux-aiml/week4-task15/src/models/proctor_model.pkl"):
    # 1. Filter to reviewed only (ground truth not null)
    df_labeled = df[df["ground_truth_label"].notna()].copy()
    
    # 2. Exclude sensor faults from training (they have NaNs in signals)
    # We will route them to no_data in inference.
    df_trainable = df_labeled.dropna(subset=["tab_switch_count", "face_count_anomalies"])
    
    df_trainable, feature_cols = engineer_features(df_trainable)
    
    # 3. Train/Val/Test split by student_id
    X = df_trainable[feature_cols + ["student_id", "scenario", "flagged_by_v0"]]
    y = df_trainable["ground_truth_label"].astype(int)
    groups = df_trainable["student_id"]
    
    # Split: 60% train, 20% val, 20% test
    gss = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups))
    
    X_train_full = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    
    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]
    
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42) # 50% of 40% = 20%
    val_idx, test_idx = next(gss_val.split(X_temp, y_temp, groups_temp))
    
    X_val_full = X_temp.iloc[val_idx]
    y_val = y_temp.iloc[val_idx]
    
    X_test_full = X_temp.iloc[test_idx]
    y_test = y_temp.iloc[test_idx]
    
    # 4. Train Model
    X_train = X_train_full[feature_cols].fillna(0)
    X_val = X_val_full[feature_cols].fillna(0)
    X_test = X_test_full[feature_cols].fillna(0)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    # 5. Tune Threshold on Validation Set
    y_val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_fpr = 1.0
    
    for t in np.arange(0.1, 0.9, 0.05):
        y_val_pred = (y_val_probs >= t).astype(int)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Require recall >= 0.80
        if recall >= 0.80 and fpr < best_fpr:
            best_fpr = fpr
            best_threshold = t
            
    print(f"Chosen threshold: {best_threshold:.2f} (Val FPR: {best_fpr:.4f})")
    
    # 6. Evaluate on Test Set
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= best_threshold).astype(int)
    
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Baseline for Test Set
    y_test_baseline = X_test_full["flagged_by_v0"]
    base_precision = precision_score(y_test, y_test_baseline, zero_division=0)
    base_recall = recall_score(y_test, y_test_baseline, zero_division=0)
    btn, bfp, bfn, btp = confusion_matrix(y_test, y_test_baseline).ravel()
    base_fpr = bfp / (bfp + btn) if (bfp + btn) > 0 else 0.0
    
    print("\n--- Test Set Evaluation ---")
    print(f"Baseline -> FPR: {base_fpr:.4f}, Recall: {base_recall:.4f}, FPs: {bfp}")
    print(f"Model    -> FPR: {fpr:.4f}, Recall: {recall:.4f}, FPs: {fp}")
    if fpr < base_fpr:
        reduction = (base_fpr - fpr) / base_fpr * 100
        print(f"[SUCCESS] FPR reduced by {reduction:.1f}%")
    else:
        print("[FAIL] FPR was not reduced.")
        
    # 7. Segment Breakdown by FP Pattern
    print("\n--- Segment Breakdown (Test Set) ---")
    results = X_test_full.copy()
    results["y_true"] = y_test
    results["y_pred_baseline"] = y_test_baseline
    results["y_pred_model"] = y_test_pred
    
    # False Positives in Baseline
    base_fps = results[(results["y_true"] == 0) & (results["y_pred_baseline"] == 1)]
    for pattern in ["fp_network", "fp_cat", "fp_copypaste", "normal_clean"]:
        pattern_fps = base_fps[base_fps["scenario"] == pattern]
        if len(pattern_fps) > 0:
            model_cleared = len(pattern_fps[pattern_fps["y_pred_model"] == 0])
            print(f"{pattern}: Baseline flagged {len(pattern_fps)} -> Model cleared {model_cleared}/{len(pattern_fps)}")

    # 8. Save Model and Metadata
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "threshold": best_threshold
    }, output_model_path)
    print(f"\nModel saved to {output_model_path}")

    return {
        "test_results": results,
        "metrics": {
            "baseline": {"precision": float(base_precision), "recall": float(base_recall), "fpr": float(base_fpr), "fp_count": int(bfp)},
            "model": {"precision": float(precision), "recall": float(recall), "fpr": float(fpr), "fp_count": int(fp)}
        },
        "threshold": float(best_threshold)
    }

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_and_validate_data
    df = load_and_validate_data()
    train_and_evaluate(df)
