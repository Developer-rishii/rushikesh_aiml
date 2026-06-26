import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import time
import os
import uuid
import csv

def evaluate_threshold(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, accuracy, f1, fpr

def calibrate(df, log_path='d:/Placemux-aiml/week3-task8/metrics/experiment_log.csv'):
    if df is None or len(df) < 50:
        raise ValueError("Dataset is too small for threshold calibration. Minimum 50 samples required.")
        
    # Ground truth: 1 = bad match (is_success==0), 0 = good match (is_success==1)
    y_true = (df['is_success'] == 0).astype(int)
    if len(y_true.unique()) < 2:
        raise ValueError("Dataset must contain both positive and negative samples for calibration.")
    
    # Sweep every 1% from 5 to 95
    thresholds = range(5, 96, 1)
    
    results = []
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Overwrite the log for a clean run
    with open(log_path, 'w', newline='') as csvfile:
        fieldnames = ['run_id', 'timestamp', 'threshold', 'precision', 'recall', 'accuracy', 'f1_score', 'false_positive_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
            
        run_id = str(uuid.uuid4())[:8]
        
        for t in thresholds:
            y_pred = (df['prediction_score'] < t).astype(int)
            p, r, a, f1, fpr = evaluate_threshold(y_true, y_pred)
            
            row = {
                'run_id': run_id,
                'timestamp': int(time.time()),
                'threshold': t,
                'precision': round(p, 4),
                'recall': round(r, 4),
                'accuracy': round(a, 4),
                'f1_score': round(f1, 4),
                'false_positive_rate': round(fpr, 4)
            }
            writer.writerow(row)
            results.append(row)
    
    # Select best threshold: maximize F1
    # Business reason: F1 balances precision and recall. For a spend-protection
    # guardrail, false negatives (letting bad matches through) cost real money,
    # while false positives (blocking good matches) cost opportunity. F1 gives
    # equal weight to both, which is the right default until the business
    # explicitly tells us to weight one side more.
    best_t = 50 
    best_f1 = -1
    for res in results:
        if res['f1_score'] > best_f1:
            best_f1 = res['f1_score']
            best_t = res['threshold']
    
    # Compute dumb baseline: warn everyone (threshold = 100)
    y_dumb = np.ones(len(y_true), dtype=int)
    dp, dr, da, df1, dfpr = evaluate_threshold(y_true, y_dumb)
    
    print("=" * 65)
    print("  THRESHOLD CALIBRATION RESULTS")
    print("=" * 65)
    print(f"  Best threshold (max F1): {best_t}%")
    print(f"  Best F1: {best_f1:.4f}")
    print()
    print(f"  {'Metric':<25} {'Dumb Baseline':>15} {'Calibrated':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    
    # Get calibrated metrics at best threshold
    y_cal = (df['prediction_score'] < best_t).astype(int)
    cp, cr, ca, cf1, cfpr = evaluate_threshold(y_true, y_cal)
    
    print(f"  {'Precision':<25} {dp:>15.4f} {cp:>15.4f}")
    print(f"  {'Recall':<25} {dr:>15.4f} {cr:>15.4f}")
    print(f"  {'Accuracy':<25} {da:>15.4f} {ca:>15.4f}")
    print(f"  {'F1 Score':<25} {df1:>15.4f} {cf1:>15.4f}")
    print(f"  {'False Positive Rate':<25} {dfpr:>15.4f} {cfpr:>15.4f}")
    print("=" * 65)
    print()
    print(f"  Business rationale: F1 maximization balances blocking bad matches")
    print(f"  (recall) against not wrongly blocking good matches (precision).")
    print(f"  The calibrated threshold improves precision by")
    print(f"  {(cp - dp)*100:+.1f}pp over the dumb baseline while maintaining")
    print(f"  {cr*100:.1f}% recall.")
    print("=" * 65)
    
    return best_t, results

if __name__ == '__main__':
    df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')
    best_t, _ = calibrate(df)
    print(f"\nFinal calibrated threshold: {best_t}%")
