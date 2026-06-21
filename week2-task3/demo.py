import pandas as pd
from src.preprocess import generate_data
from src.feature_engineering import compute_features
from src.baseline import calculate_baseline_score, evaluate_baseline
from src.train_model import train_logistic_regression
from src.evaluation import evaluate_model
from src.ranking import Ranker
import json
import os

def run_demo():
    print("="*50)
    print("1. Data Generation")
    print("="*50)
    # Generate smaller dataset for fast demo if needed, but we already have data
    if not os.path.exists("data/students.csv"):
        generate_data(num_students=1000, num_jobs=200, num_applications=5000)
    else:
        print("Data already generated.")
        
    print("\n" + "="*50)
    print("2. Feature Engineering & Baseline")
    print("="*50)
    if not os.path.exists("data/baseline_results.csv"):
        df_students = pd.read_csv("data/students.csv")
        df_jobs = pd.read_csv("data/jobs.csv")
        df_apps = pd.read_csv("data/applications.csv")
        df_features = compute_features(df_students, df_jobs, df_apps)
        df_features.to_csv("data/features.csv", index=False)
        df_scored = calculate_baseline_score(df_features)
        evaluate_baseline(df_scored)
        df_scored.to_csv("data/baseline_results.csv", index=False)
    else:
        print("Baseline already evaluated.")
        
    print("\n" + "="*50)
    print("3. Model Training & Evaluation")
    print("="*50)
    if not os.path.exists("models/model.pkl"):
        train_logistic_regression()
    else:
        print("Model already trained.")
        
    evaluate_model()

    print("\n" + "="*50)
    print("4. Demonstration: Ranking & Explainability")
    print("="*50)
    
    ranker = Ranker()
    
    # Get a sample student
    sample_student_id = "S00001"
    print(f"\n--- Ranking Jobs for Student {sample_student_id} ---")
    jobs_ranked = ranker.rank_jobs_for_student(sample_student_id, top_n=3)
    print(json.dumps(jobs_ranked, indent=2))
    
    # Get a sample job
    sample_job_id = "J00001"
    print(f"\n--- Ranking Candidates for Job {sample_job_id} ---")
    candidates_ranked = ranker.rank_candidates_for_job(sample_job_id, top_n=3)
    print(json.dumps(candidates_ranked, indent=2))

if __name__ == "__main__":
    run_demo()
