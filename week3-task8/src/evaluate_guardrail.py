import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from src.threshold_calibration import calibrate, evaluate_threshold

def main():
    df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')
    
    # 70/15/15 split
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    
    train_val_df = pd.concat([train, val])
    
    print(f"Train+Val size: {len(train_val_df)}, Test size: {len(test)}")
    print()
    
    # Calibrate on train+val
    print("--- CALIBRATION (on train+val) ---")
    best_threshold, _ = calibrate(train_val_df)
    print()
    
    # Evaluate on TEST set
    print("=" * 65)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 65)
    
    y_true = (test['is_success'] == 0).astype(int)
    
    # Dumb baseline on test
    y_dumb = np.ones(len(y_true), dtype=int)
    dp, dr, da, df1, dfpr = evaluate_threshold(y_true, y_dumb)
    
    # Calibrated guardrail on test
    y_pred = (test['prediction_score'] < best_threshold).astype(int)
    cp, cr, ca, cf1, cfpr = evaluate_threshold(y_true, y_pred)
    
    print(f"  Threshold used: {best_threshold}%")
    print()
    print(f"  {'Metric':<25} {'Dumb Baseline':>15} {'Calibrated':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Precision':<25} {dp:>15.4f} {cp:>15.4f}")
    print(f"  {'Recall':<25} {dr:>15.4f} {cr:>15.4f}")
    print(f"  {'Accuracy':<25} {da:>15.4f} {ca:>15.4f}")
    print(f"  {'F1 Score':<25} {df1:>15.4f} {cf1:>15.4f}")
    print(f"  {'False Positive Rate':<25} {dfpr:>15.4f} {cfpr:>15.4f}")
    print("=" * 65)
    
    metrics = {
        "precision": round(cp, 4),
        "recall": round(cr, 4),
        "accuracy": round(ca, 4),
        "f1_score": round(cf1, 4),
        "false_positive_rate": round(cfpr, 4),
        "threshold_used": best_threshold,
        "dumb_baseline": {
            "precision": round(dp, 4),
            "recall": round(dr, 4),
            "accuracy": round(da, 4),
            "f1_score": round(df1, 4),
            "false_positive_rate": round(dfpr, 4)
        }
    }
    
    os.makedirs('d:/Placemux-aiml/week3-task8/metrics', exist_ok=True)
    with open('d:/Placemux-aiml/week3-task8/metrics/guardrail_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nMetrics saved to metrics/guardrail_metrics.json")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()
