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

def is_true_match(student: Dict[str, Any], job: Dict[str, Any]) -> int:
    """
    Principled ground-truth definition checking if student meets ALL job minimums.
    """
    # 1. average of their relevant skill scores >= job's average minimum skill score
    job_skills = job.get('Skill Requirements', {})
    if job_skills:
        avg_job_skill = sum(job_skills.values()) / len(job_skills)
        student_skills = student.get('Skills', {})
        relevant_student_scores = [student_skills.get(s, 0) for s in job_skills.keys()]
        avg_student_skill = sum(relevant_student_scores) / len(job_skills)
        if avg_student_skill < avg_job_skill:
            return 0
            
    # 2. CGPA >= job's minimum CGPA
    if student.get('CGPA', 0.0) < job.get('Minimum CGPA', 0.0):
        return 0
        
    # 3. experience units >= job's requirement
    req_exp = job.get('Experience Requirement', 0)
    projects = student.get('Project Count', 0)
    internships = student.get('Internship Count', 0)
    student_exp_units = internships + (projects * 0.5)
    if student_exp_units < req_exp:
        return 0
        
    return 1

def simulate_ground_truth_and_evaluate(matcher_func, students_df, jobs_df):
    """
    Creates synthetic ground truth for demo purposes, runs evaluation, and logs it.
    """
    y_true = []
    y_pred_scores = []
    
    for _, job_row in jobs_df.iterrows():
        job_id = job_row['Job ID']
        job_dict = job_row.to_dict()
        
        for _, student_row in students_df.iterrows():
            student_id = student_row['Student ID']
            student_dict = student_row.to_dict()
            
            # Ground truth
            truth_label = is_true_match(student_dict, job_dict)
            y_true.append(truth_label)
            
            # Predicted
            score, _ = matcher_func(student_dict, job_dict)
            y_pred_scores.append(score)
            
    pos_rate = sum(y_true) / len(y_true) if y_true else 0.0
    print(f"Ground Truth Positive Rate: {pos_rate:.2%} ({sum(y_true)}/{len(y_true)} true matches)")
    
    metrics = evaluate_system(y_true, y_pred_scores, threshold=75)
    
    tracker = ExperimentTracker()
    tracker.log_experiment("baseline_rule_based", "v1", metrics)
    
    return metrics
