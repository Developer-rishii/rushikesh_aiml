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
    
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
    return precision, recall, accuracy, f1, fpr

def calibrate(df, log_path='d:/Placemux-aiml/week3-task8/metrics/experiment_log.csv'):
    if df is None or len(df) < 50:
        raise ValueError("Dataset is too small for threshold calibration. Minimum 50 samples required.")
        
    y_true = (df['is_success'] == 0).astype(int)
    if len(y_true.unique()) < 2:
        raise ValueError("Dataset must contain both positive and negative samples for calibration.")
        
    thresholds = range(10, 91, 5) # Sweep from 10 to 90
    
    results = []
    
    y_pred_base = (df['prediction_score'] < 50).astype(int)
    bp, br, ba, bf1, bfpr = evaluate_threshold(y_true, y_pred_base)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    
    with open(log_path, 'a', newline='') as csvfile:
        fieldnames = ['run_id', 'timestamp', 'threshold', 'precision', 'recall', 'accuracy', 'f1_score', 'false_positive_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        run_id = str(uuid.uuid4())
        
        for t in thresholds:
            y_pred = (df['prediction_score'] < t).astype(int)
            p, r, a, f1, fpr = evaluate_threshold(y_true, y_pred)
            
            row = {
                'run_id': run_id,
                'timestamp': int(time.time()),
                'threshold': t,
                'precision': p,
                'recall': r,
                'accuracy': a,
                'f1_score': f1,
                'false_positive_rate': fpr
            }
            writer.writerow(row)
            results.append(row)
            
    best_t = 50 
    best_score = -1
    for res in results:
        score = res['f1_score'] - res['false_positive_rate']
        if score > best_score:
            best_score = score
            best_t = res['threshold']
            
    return best_t, results

if __name__ == '__main__':
    df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')
    best_t, _ = calibrate(df)
    print(f"Calibrated best threshold: {best_t}")
