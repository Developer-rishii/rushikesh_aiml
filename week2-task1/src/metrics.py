from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class ExperimentTracker:
    def __init__(self, log_file="experiments.jsonl"):
        self.log_file = log_file

    def log_experiment(self, name: str, dataset_version: str, metrics: Dict[str, float]):
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": name,
            "dataset_version": dataset_version,
            "metrics": metrics
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Logged experiment: {name} - {metrics}")

def evaluate_system(y_true: List[int], y_pred_scores: List[int], threshold: int = 75) -> Dict[str, float]:
    """
    Evaluates the matching engine using ground truth data.
    y_true: List of binary labels (1 for true match, 0 for not a match)
    y_pred_scores: List of predicted match scores (0-100)
    threshold: Score above which a match is considered positive
    """
    y_pred = [1 if score >= threshold else 0 for score in y_pred_scores]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0 # Edge case if all true or all false
        
    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "accuracy": float(accuracy)
    }

def simulate_ground_truth_and_evaluate(matcher_func, students_df, jobs_df):
    """
    Creates synthetic ground truth for demo purposes, runs evaluation, and logs it.
    """
    y_true = []
    y_pred_scores = []
    
    # Simulate some ground truth: We'll just define a naive rule for "true" match
    # to test the metrics pipeline. E.g. If job 101, student 1 and 3 are true matches.
    true_matches = {
        101: [1, 3], # TechCorp Data Scientist -> Alice, Charlie
        104: [1, 2, 3, 5, 8], # Python Dev -> Many have Python
        110: [1, 2, 3, 4, 5, 6, 8, 9, 10] # SQL Analyst -> Most have SQL
    }
    
    for _, job_row in jobs_df.iterrows():
        job_id = job_row['Job ID']
        job_dict = job_row.to_dict()
        
        for _, student_row in students_df.iterrows():
            student_id = student_row['Student ID']
            student_dict = student_row.to_dict()
            
            # Ground truth
            is_true_match = 1 if student_id in true_matches.get(job_id, []) else 0
            y_true.append(is_true_match)
            
            # Predicted
            score, _ = matcher_func(student_dict, job_dict)
            y_pred_scores.append(score)
            
    metrics = evaluate_system(y_true, y_pred_scores, threshold=75)
    
    tracker = ExperimentTracker()
    tracker.log_experiment("baseline_rule_based", "v1", metrics)
    
    return metrics
