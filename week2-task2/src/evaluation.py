import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import precision_score, recall_score, confusion_matrix
from src.match_vector import generate_match_vector
from src.threshold_validator import validate_thresholds

def run_evaluation():
    print("Running evaluation...")
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    students_path = os.path.join(base_dir, "data", "students.csv")
    jobs_path = os.path.join(base_dir, "data", "jobs.csv")
    
    if not os.path.exists(students_path) or not os.path.exists(jobs_path):
        print("Data files not found. Run 'python src/load_data.py' first.")
        return
        
    students_df = pd.read_csv(students_path)
    jobs_df = pd.read_csv(jobs_path)
    
    # Generate some semi-synthetic "true labels" for evaluation.
    # In a real system, these would be provided by human reviewers.
    # Here, we'll create a slightly noisy version of our own logic to simulate "ground truth".
    # We match all students against the first job for this evaluation.
    
    job_idx = 0
    job_dict = jobs_df.iloc[job_idx].to_dict()
    job_id = job_dict.pop('job_id')
    
    y_true = []
    y_pred = []
    
    for i in range(len(students_df)):
        student_dict = students_df.iloc[i].to_dict()
        student_dict.pop('student_id')
        
        # System prediction
        validation = validate_thresholds(student_dict, job_dict)
        predicted_eligibility = 1 if validation['eligible'] else 0
        y_pred.append(predicted_eligibility)
        
        # Simulated "Ground Truth" (e.g. human might occasionally override a slight miss)
        # We will add 5% random flip noise to simulate subjective human labels
        import random
        random.seed(i)
        if random.random() < 0.05:
            y_true.append(1 - predicted_eligibility)
        else:
            y_true.append(predicted_eligibility)
            
    # Calculate Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        # Edge case if only one class exists
        fpr = 0.0
        
    metrics = {
        "job_id": [job_id],
        "precision": [round(precision, 4)],
        "recall": [round(recall, 4)],
        "false_positive_rate": [round(fpr, 4)],
        "true_positives": [tp if cm.shape == (2,2) else 0],
        "false_positives": [fp if cm.shape == (2,2) else 0],
        "true_negatives": [tn if cm.shape == (2,2) else 0],
        "false_negatives": [fn if cm.shape == (2,2) else 0]
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    # Save to logs
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logs_path = os.path.join(logs_dir, "experiments.csv")
    
    # Append or write new
    if os.path.exists(logs_path):
        metrics_df.to_csv(logs_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(logs_path, index=False)
        
    print("Evaluation complete.")
    print(metrics_df.to_string(index=False))
    print(f"Logs saved to {logs_path}")

if __name__ == "__main__":
    run_evaluation()
