import os
import csv
from datetime import datetime

def log_experiment(metrics, notes, log_file="experiments/experiment_log.csv"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        fieldnames = [
            'timestamp', 'model', 'accuracy', 'precision', 'recall', 'f1_score',
            'false_positive_rate', 'false_negative_rate', 'notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': 'LogisticRegression',
            'accuracy': f"{metrics.get('Accuracy', 0)*100:.2f}%",
            'precision': f"{metrics.get('Precision', 0)*100:.2f}%",
            'recall': f"{metrics.get('Recall', 0)*100:.2f}%",
            'f1_score': f"{metrics.get('F1 Score', 0)*100:.2f}%",
            'false_positive_rate': f"{metrics.get('False Positive Rate', 0)*100:.2f}%",
            'false_negative_rate': f"{metrics.get('False Negative Rate', 0)*100:.2f}%",
            'notes': notes
        }
        writer.writerow(row)
