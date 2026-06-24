import os
import json
import csv
import pickle
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from data_loader import load_data
from feature_engineering import extract_features

def evaluate_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    models_dir = os.path.join(current_dir, "..", "models")
    metrics_dir = os.path.join(current_dir, "..", "metrics")
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "baseline_model.pkl")
    try:
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
            model = artifact["model"]
            scaler = artifact["scaler"]
    except (FileNotFoundError, pickle.UnpicklingError, KeyError) as e:
        raise ValueError(f"Failed to load valid model from {model_path}. Error: {str(e)}")
        
    candidates_df, jobs_df = load_data(data_dir)
    features_df = extract_features(candidates_df, jobs_df)
    
    X = features_df[['skill_overlap_percentage', 'experience_gap', 'education_match', 'certification_match_count', 'required_skill_coverage']]
    y = features_df['label']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
        
    metrics = {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "accuracy": float(round(accuracy, 4)),
        "f1_score": float(round(f1, 4)),
        "false_positive_rate": float(round(fpr, 4))
    }
    
    metrics_json_path = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    log_path = os.path.join(metrics_dir, "experiment_log.csv")
    file_exists = os.path.exists(log_path)
    
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()
    
    with open(log_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_id", "timestamp", "model", "precision", "recall", "accuracy", "f1_score", "false_positive_rate"])
        writer.writerow([run_id, timestamp, "LogisticRegression", metrics['precision'], metrics['recall'], metrics['accuracy'], metrics['f1_score'], metrics['false_positive_rate']])
        
    print(f"Evaluation complete. Metrics saved to {metrics_json_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    evaluate_model()
