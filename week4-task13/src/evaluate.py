import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, precision_recall_curve, average_precision_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.baseline import baseline_rule

def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return precision, recall, fpr, f1

def evaluate():
    test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_set.csv')
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
    
    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print("Test data or model not found. Run train_model.py first.")
        return
        
    df_test = pd.read_csv(test_path)
    
    # Reconstruct original features for baseline since it needs them
    # Wait, df_test has them except it has 'true_violation' instead of 'label'
    df_test['label'] = df_test['true_violation'].apply(lambda x: 'true_violation' if x == 1 else 'false_positive')
    X_test = df_test.drop(columns=['true_violation', 'label'])
    y_true = df_test['true_violation'].values
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    y_pred_model = model.predict(X_test)
    y_prob_model = model.predict_proba(X_test)[:, 1]
    
    y_pred_base = df_test.apply(baseline_rule, axis=1)
    
    # Global Metrics
    p_base, r_base, fpr_base, f1_base = get_metrics(y_true, y_pred_base)
    p_mod, r_mod, fpr_mod, f1_mod = get_metrics(y_true, y_pred_model)
    
    print("\n" + "="*40)
    print("GLOBAL METRICS (HELD-OUT TEST SET)")
    print("="*40)
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Model':<10}")
    print("-" * 40)
    print(f"{'Precision':<15} | {p_base:<10.3f} | {p_mod:<10.3f}")
    print(f"{'Recall':<15} | {r_base:<10.3f} | {r_mod:<10.3f}")
    print(f"{'FPR':<15} | {fpr_base:<10.3f} | {fpr_mod:<10.3f}")
    print(f"{'F1 Score':<15} | {f1_base:<10.3f} | {f1_mod:<10.3f}")
    
    # FP Reduction
    fp_base_count = confusion_matrix(y_true, y_pred_base, labels=[0, 1]).ravel()[1]
    fp_mod_count = confusion_matrix(y_true, y_pred_model, labels=[0, 1]).ravel()[1]
    
    if fp_base_count > 0:
        fp_reduction = ((fp_base_count - fp_mod_count) / fp_base_count) * 100
        print(f"\n=> False Positive Reduction Achieved: {fp_reduction:.1f}%")
    else:
        print("\n=> False Positive Reduction Achieved: N/A (Baseline had 0 FPs)")
        
    print("\n" + "="*40)
    print("SEGMENT BREAKDOWN (FPR & Recall)")
    print("="*40)
    
    # Define segments
    segments = {
        'Device: Desktop': df_test['device_type'] == 'desktop',
        'Device: Mobile': df_test['device_type'] == 'mobile',
        'Device: Laptop': df_test['device_type'] == 'laptop',
        'Network: Poor': df_test['network_quality'] == 'poor',
        'Network: Excellent': df_test['network_quality'] == 'excellent',
        'Flag: High Tab Switches': df_test['tab_switch_count'] > 3,
        'Flag: Face Absent': df_test['face_absent_seconds'] > 10
    }
    
    for seg_name, mask in segments.items():
        if mask.sum() == 0:
            continue
        y_t_seg = y_true[mask]
        if len(np.unique(y_t_seg)) < 2:
            continue # Need both classes
            
        y_p_b_seg = y_pred_base[mask]
        y_p_m_seg = y_pred_model[mask]
        
        _, r_b, fpr_b, _ = get_metrics(y_t_seg, y_p_b_seg)
        _, r_m, fpr_m, _ = get_metrics(y_t_seg, y_p_m_seg)
        
        print(f"--- {seg_name} ---")
        print(f"  Baseline -> FPR: {fpr_b:.3f}, Recall: {r_b:.3f}")
        print(f"  Model    -> FPR: {fpr_m:.3f}, Recall: {r_m:.3f}")

    # Plot PR Curve
    pr_base, rec_base, _ = precision_recall_curve(y_true, y_pred_base) # discrete
    pr_mod, rec_mod, _ = precision_recall_curve(y_true, y_prob_model)
    ap_mod = average_precision_score(y_true, y_prob_model)
    
    plt.figure(figsize=(8,6))
    plt.plot(rec_mod, pr_mod, label=f'Model (AP={ap_mod:.3f})', color='blue')
    # Baseline is a single point, so let's plot it as a point
    plt.plot(r_base, p_base, marker='x', markersize=10, color='red', label='Baseline Rule', linestyle='None')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Model vs Baseline)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'demo', 'pr_curve.png')
    plt.savefig(plot_path)
    print(f"\nSaved PR Curve plot to {plot_path}")

if __name__ == "__main__":
    evaluate()
