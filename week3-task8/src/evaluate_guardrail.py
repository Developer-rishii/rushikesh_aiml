import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from .threshold_calibration import calibrate, evaluate_threshold

def main():
    df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')
    
    # 70/15/15 split
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    
    train_val_df = pd.concat([train, val])
    
    # Calibrate on train+val
    best_threshold, _ = calibrate(train_val_df)
    
    # Evaluate on test
    y_true = (test['is_success'] == 0).astype(int)
    y_pred = (test['prediction_score'] < best_threshold).astype(int)
    
    p, r, a, f1, fpr = evaluate_threshold(y_true, y_pred)
    
    metrics = {
        "precision": p,
        "recall": r,
        "accuracy": a,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "threshold_used": best_threshold
    }
    
    os.makedirs('d:/Placemux-aiml/week3-task8/metrics', exist_ok=True)
    with open('d:/Placemux-aiml/week3-task8/metrics/guardrail_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to metrics/guardrail_metrics.json:")
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    main()
