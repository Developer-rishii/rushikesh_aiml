import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score
import joblib
import json
import os

def prepare_data(features_df: pd.DataFrame):
    df = features_df[features_df['response_count'] >= 20].copy()
    
    # Encode categorical features
    df = pd.get_dummies(df, columns=['subject'], drop_first=False)
    
    # Ensure stable column names for categories
    feature_cols = [c for c in df.columns if c not in ['item_id', 'is_weak_item', 'split']]
    
    train = df[df['split'] == 'train']
    val = df[df['split'] == 'val']
    test = df[df['split'] == 'test']
    
    X_train = train[feature_cols]
    y_train = train['is_weak_item']
    
    X_val = val[feature_cols]
    y_val = val['is_weak_item']
    
    X_test = test[feature_cols]
    y_test = test['is_weak_item']
    
    # We return test to compute segmented metrics later
    test_original = features_df[(features_df['split'] == 'test') & (features_df['response_count'] >= 20)].copy()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, test_original, feature_cols

def baseline_predict(test_df: pd.DataFrame) -> np.ndarray:
    return (test_df['p_value'] < 0.05) | (test_df['p_value'] > 0.95)

def train_and_evaluate(features_df: pd.DataFrame, model_dir: str, reports_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test, test_df, feature_cols = prepare_data(features_df)
    
    X_cv = pd.concat([X_train, X_val])
    y_cv = pd.concat([y_train, y_val])
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_cv, y_cv)
    
    best_model = grid_search.best_estimator_
    
    joblib.dump(best_model, os.path.join(model_dir, "weak_item_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    y_pred_base = baseline_predict(test_df)
    
    def compute_metrics(y_true, y_p, y_prob=None):
        if len(y_true) == 0:
            return None
        # Handle cases with only one class in y_true
        if len(np.unique(y_true)) == 1:
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_p, labels=[False, True]).ravel()
                precision = precision_score(y_true, y_p, labels=[False, True], zero_division=0)
                recall = recall_score(y_true, y_p, labels=[False, True], zero_division=0)
            except:
                precision, recall, fpr, auc = 0, 0, 0, 0
                return {"precision": 0.0, "recall": 0.0, "fpr": 0.0, "auc": 0.0}
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
            precision = precision_score(y_true, y_p, zero_division=0)
            recall = recall_score(y_true, y_p, zero_division=0)
            
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else None
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "auc": round(auc, 4) if auc is not None else None
        }
        
    metrics = {
        "model": compute_metrics(y_test, y_pred, y_pred_proba),
        "baseline": compute_metrics(y_test, y_pred_base, y_pred_base.astype(float))
    }
    
    subject_metrics = {}
    for subj in test_df['subject'].unique():
        idx = test_df['subject'] == subj
        if idx.sum() > 0:
            subject_metrics[subj] = compute_metrics(y_test[idx], y_pred[idx], y_pred_proba[idx])
            
    buckets = [
        (20, 50, "20-50"),
        (51, 150, "51-150"),
        (151, 99999, "151+")
    ]
    bucket_metrics = {}
    for lower, upper, label in buckets:
        idx = (test_df['response_count'] >= lower) & (test_df['response_count'] <= upper)
        if idx.sum() > 0:
            bucket_metrics[label] = compute_metrics(y_test[idx], y_pred[idx], y_pred_proba[idx])
            
    metrics["segments"] = {
        "subject": subject_metrics,
        "response_count": bucket_metrics
    }
    
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    report_md = f"""# PlaceMux Task 19 Sign-Off Report

## Model Evaluation
The ML model was evaluated on a strictly held-out item-level test set, preventing label leakage. Items with fewer than 20 responses were excluded from metrics to ensure reliable ground truth representation.

### Baseline vs Model Performance

| Metric | Baseline | ML Model |
|---|---|---|
| Precision | {metrics['baseline']['precision']} | {metrics['model']['precision']} |
| Recall | {metrics['baseline']['recall']} | {metrics['model']['recall']} |
| FPR | {metrics['baseline']['fpr']} | {metrics['model']['fpr']} |
| AUC | N/A | {metrics['model']['auc']} |

"""
    with open(os.path.join(reports_dir, "sign_off_report.md"), "w") as f:
        f.write(report_md)
        
    return metrics
