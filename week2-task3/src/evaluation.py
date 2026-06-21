import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(test_data_path="data/test_set.csv", model_path="models/model.pkl", log_path="experiments/experiment_log.csv"):
    if not os.path.exists(test_data_path) or not os.path.exists(model_path):
        print("Test data or model not found.")
        return
        
    df = pd.read_csv(test_data_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    features = ['skill_overlap', 'experience_match', 'education_match', 'location_match']
    X_test = df[features]
    y_true = df['successful_match']
    
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract True Negatives and False Positives for FPR
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        # edge case if only one class is present
        fpr = 0.0
        
    print("--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"False Pos Rate: {fpr:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save to experiments log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    log_data = pd.DataFrame([{
        "model": "LogisticRegression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "confusion_matrix": str(cm.tolist())
    }])
    
    if os.path.exists(log_path):
        existing_log = pd.read_csv(log_path)
        log_data = pd.concat([existing_log, log_data], ignore_index=True)
        
    log_data.to_csv(log_path, index=False)
    print(f"Results appended to {log_path}")

if __name__ == "__main__":
    evaluate_model()
