import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd
from data.generate_synthetic_sessions import load_data

def baseline_rule(row):
    """
    Dumb rule-based baseline: 
    flag = violation if tab_switch_count > 3 OR face_absent_seconds > 10
    """
    # Handle missing values gracefully
    tab_switches = row['tab_switch_count'] if pd.notna(row['tab_switch_count']) else 0
    face_absent = row['face_absent_seconds'] if pd.notna(row['face_absent_seconds']) else 0
    
    if tab_switches > 3 or face_absent > 10:
        return 1 # true_violation
    return 0 # false_positive

def evaluate_baseline(df):
    """
    Evaluate the baseline on the dataset
    """
    # Convert label to binary: true_violation=1, false_positive=0
    y_true = (df['label'] == 'true_violation').astype(int)
    y_pred = df.apply(baseline_rule, axis=1)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return precision, recall, fpr

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_sessions.csv')
    try:
        df = load_data(data_path)
        p, r, fpr = evaluate_baseline(df)
        print(f"--- Baseline Performance ---")
        print(f"Precision: {p:.3f}")
        print(f"Recall:    {r:.3f}")
        print(f"FPR:       {fpr:.3f}")
        print("----------------------------")
    except Exception as e:
        print(f"Error: {e}")
