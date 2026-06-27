import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import sqlite3
import json
import datetime
import os

DB_PATH = 'run_history.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            metrics JSON,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_run(metrics, status):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO run_history (timestamp, metrics, status)
        VALUES (?, ?, ?)
    ''', (datetime.datetime.now().isoformat(), json.dumps(metrics), status))
    conn.commit()
    conn.close()

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(y_true) > 0 else (0,0,0,0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'fpr': fpr}

def run_conversion_quality_check(df, model, baseline_metrics, threshold=0.05):
    """
    Computes metrics segmented by payment status.
    Compares paid vs. unpaid (failed/pending/refunded).
    Flags relevance regression if any segment drops below baseline precision by > threshold.
    """
    print("\n--- Conversion-Quality Check ---")
    
    # Predict using the model
    # We expect X features to be engineered, but this function takes df directly to match the pipeline
    from src.model import engineer_features
    X = engineer_features(df)
    y_pred = model.predict(X)
    df['predicted'] = y_pred
    
    segments = df['payment_status'].unique()
    segment_metrics = {}
    
    for seg in segments:
        seg_df = df[df['payment_status'] == seg]
        m = compute_metrics(seg_df['is_good_match'], seg_df['predicted'])
        segment_metrics[seg] = m
        print(f"Segment [{seg}]: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, FPR={m['fpr']:.3f}")
        
    # Check 1: Paid vs Unpaid relevance skew (for similar skill profiles)
    # To do this simply, we compare overall metrics of paid vs (failed + pending)
    paid_df = df[df['payment_status'] == 'paid']
    unpaid_df = df[df['payment_status'].isin(['failed', 'pending'])]
    
    paid_m = compute_metrics(paid_df['is_good_match'], paid_df['predicted'])
    unpaid_m = compute_metrics(unpaid_df['is_good_match'], unpaid_df['predicted'])
    
    print("\n[Skew Check] Paid vs Unpaid (Failed/Pending):")
    print(f"Paid   - Precision: {paid_m['precision']:.3f}, Recall: {paid_m['recall']:.3f}")
    print(f"Unpaid - Precision: {unpaid_m['precision']:.3f}, Recall: {unpaid_m['recall']:.3f}")
    
    # Check 2: Regression against baseline
    baseline_precision = baseline_metrics['precision']
    
    status = "PASS"
    reasons = []
    
    for seg, m in segment_metrics.items():
        if baseline_precision - m['precision'] > threshold:
            status = "FAIL"
            reasons.append(f"Segment '{seg}' precision ({m['precision']:.3f}) dropped below baseline ({baseline_precision:.3f}) by > {threshold*100}%")
            
    if status == "PASS":
        print(f"\n=> RESULT: PASS (No segment regressed beyond {threshold*100}% threshold)")
    else:
        print("\n=> RESULT: FAIL")
        for r in reasons:
            print(f"   - {r}")
            
    # Persist log
    log_run(segment_metrics, status)
    
    return status, segment_metrics
