import os
import random
import pandas as pd
from src.match_vector import generate_match_vector
from src.threshold_validator import validate_thresholds
from src.scoring import calculate_match_score
from src.explainability import generate_explanation

def run_demo():
    print("--- Job Matching System Demo ---")
    
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    students_path = os.path.join(base_dir, "data", "students.csv")
    jobs_path = os.path.join(base_dir, "data", "jobs.csv")
    
    if not os.path.exists(students_path) or not os.path.exists(jobs_path):
        print("Data files not found. Please run 'python src/load_data.py' first.")
        return
        
    students_df = pd.read_csv(students_path)
    jobs_df = pd.read_csv(jobs_path)
    
    # Pick a random student and job
    student_idx = random.randint(0, len(students_df) - 1)
    job_idx = random.randint(0, len(jobs_df) - 1)
    
    student_dict = students_df.iloc[student_idx].to_dict()
    job_dict = jobs_df.iloc[job_idx].to_dict()
    
    student_id = student_dict.pop('student_id')
    job_id = job_dict.pop('job_id')
    
    print(f"\nStudent ID: {student_id}")
    print(f"Job ID: {job_id}")
    
    # Process
    match_vector = generate_match_vector(student_dict, job_dict)
    validation = validate_thresholds(student_dict, job_dict)
    score = calculate_match_score(match_vector)
    
    print(f"\nMatch Vector:\n{match_vector}")
    print(f"\nMatch Score:\n{score}%")
    
    if validation['missing_skills']:
        print(f"\nMissing Skills:\n{', '.join(validation['missing_skills'])}")
    else:
        print(f"\nMissing Skills:\nNone")
        
    print(f"\nEligible:\n{'Yes' if validation['eligible'] else 'No'}")
    
    # Simple reason text (similar to the requirements example)
    total_reqs = len(job_dict)
    met_reqs = sum(match_vector)
    print(f"\nReason:\nCandidate met {met_reqs} of {total_reqs} required thresholds.")
    
    print("\n--- Detailed Explanation ---")
    explanation = generate_explanation(student_dict, job_dict, score, validation['eligible'], match_vector)
    print(explanation)

if __name__ == "__main__":
    run_demo()
