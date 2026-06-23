import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import ast
from baseline_matcher import BaselineMatcher
from ranker import CandidateRanker
from ml_model import create_applications_dataset, feature_engineering

# Ensure the output directory exists
os.makedirs("docs/demo_evidence", exist_ok=True)

def generate_evidence():
    print("Loading data...")
    jobs_df = pd.read_csv("data/jobs.csv")
    students_df = pd.read_csv("data/students.csv")
    
    # 1. Sample dataset statistics
    stats = {
        "Total Jobs": len(jobs_df),
        "Total Students": len(students_df),
        "Average Required Skills per Job": jobs_df['required_skills'].apply(lambda x: len(str(x).split(','))).mean(),
        "Average Verified Skills per Student": students_df['verified_skills'].apply(lambda x: len(str(x).split(','))).mean()
    }
    
    with open("docs/demo_evidence/dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Generating application pairs...")
    apps_df = create_applications_dataset(jobs_df, students_df)
    
    # Compute Match Scores using BaselineMatcher
    matcher = BaselineMatcher(threshold=70.0)
    scores = []
    statuses = []
    
    print("Calculating baseline matches for distribution...")
    for _, row in apps_df.iterrows():
        # Reconstruct series format
        job_s = pd.Series({'job_id': row['job_id'], 'required_skills': row['required_skills'], 'experience_required': row['experience_required'], 'minimum_skill_score': row['minimum_skill_score']})
        student_s = pd.Series({'student_id': row['student_id'], 'verified_skills': row['verified_skills'], 'experience': row['experience'], 'skill_scores': row['skill_scores']})
        
        res = matcher.match(job_s, student_s)
        if "error" not in res:
            scores.append(res['match_score'])
            statuses.append(res['status'])
            
    # 2. Match distribution charts
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, color='blue')
    plt.title('Distribution of Match Scores')
    plt.xlabel('Match Score (%)')
    plt.ylabel('Frequency')
    plt.axvline(x=70, color='red', linestyle='--', label='Threshold (70%)')
    plt.legend()
    plt.savefig("docs/demo_evidence/match_distribution.png")
    plt.close()
    
    # 3. Threshold pass/fail distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=statuses, palette={'eligible': 'green', 'rejected': 'red'})
    plt.title('Threshold Pass/Fail Distribution')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.savefig("docs/demo_evidence/threshold_distribution.png")
    plt.close()
    
    # 4. Candidate ranking examples
    print("Generating candidate ranking examples...")
    ranker = CandidateRanker(matcher)
    sample_job = jobs_df.iloc[0]
    ranked = ranker.rank_candidates(sample_job, students_df)
    
    ranking_example = {
        "Job ID": str(sample_job['job_id']),
        "Job Title": str(sample_job['title']),
        "Required Skills": str(sample_job['required_skills']),
        "Top 5 Candidates": []
    }
    
    for i, c in enumerate(ranked[:5]):
        ranking_example["Top 5 Candidates"].append({
            "Rank": i + 1,
            "Student ID": c['student_id'],
            "Match Score": c['match_score'],
            "Reasoning": c['reasons_list']
        })
        
    with open("docs/demo_evidence/ranking_example.json", "w") as f:
        json.dump(ranking_example, f, indent=4)
        
    print("Evidence generated in docs/demo_evidence/")

if __name__ == "__main__":
    generate_evidence()
