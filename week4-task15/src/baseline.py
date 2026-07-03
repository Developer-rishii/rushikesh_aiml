import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def compute_baseline_metrics(df):
    # Only evaluate on rows with ground truth
    df_eval = df[df["ground_truth_label"].notna()].copy()
    
    y_true = df_eval["ground_truth_label"]
    y_pred = df_eval["flagged_by_v0"]

    # Filter out sensor faults from evaluation if they can't be scored by v0
    # Wait, the v0 flag might be NaN for sensor faults. Let's drop NaN predictions for baseline
    valid_mask = y_pred.notna()
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    precision = precision_score(y_true_valid, y_pred_valid, zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("--- v0 Baseline Metrics ---")
    print(f"Evaluated on {len(y_true_valid)} labeled sessions")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Absolute False Positives: {fp}")

    return {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fp_count": fp,
        "total_evaluated": len(y_true_valid)
    }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_and_validate_data
    
    df = load_and_validate_data()
    compute_baseline_metrics(df)
