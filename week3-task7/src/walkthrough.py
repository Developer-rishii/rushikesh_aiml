import os
import json
import pandas as pd
from features import extract_features
from matcher import JobMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Load some sample data
    candidates_df = pd.read_csv(os.path.join(BASE_DIR, '../data/candidates.csv'))
    jobs_df = pd.read_csv(os.path.join(BASE_DIR, '../data/jobs.csv'))
    
    # Pick a specific candidate and job for the demo
    cand_row = candidates_df.iloc[0]
    candidate = {
        'Candidate ID': cand_row['Candidate ID'],
        'Skills': cand_row['Skills'],
        'Experience Years': cand_row['Experience Years'],
        'Education': cand_row['Education'],
        'Certifications': cand_row['Certifications'],
        'Projects': cand_row['Projects']
    }
    
    # We will pick 5 random jobs and rank them, and show details for the top ranked one.
    sample_jobs = jobs_df.sample(5, random_state=42).to_dict('records')
    
    matcher = JobMatcher(model_path=os.path.join(BASE_DIR, '../models/logistic_regression.joblib'))
    
    print("=============================================")
    print(f"CANDIDATE: {candidate['Candidate ID']}")
    print("=============================================")
    print(f"Skills:       {candidate['Skills']}")
    print(f"Experience:   {candidate['Experience Years']} years")
    print(f"Education:    {candidate['Education']}")
    print(f"Certs:        {candidate['Certifications']}")
    print(f"Projects:     {candidate['Projects']}")
    print("\nRanking against 5 sample jobs...\n")
    
    ranked_results = matcher.rank_jobs(candidate, sample_jobs)
    
    top_result = ranked_results[0]
    top_job = next(j for j in sample_jobs if j['Job ID'] == top_result['job_id'])
    
    print("=============================================")
    print(f"TOP JOB: {top_job['Job ID']}")
    print("=============================================")
    print(f"Required Skills:  {top_job['Required Skills']}")
    print(f"Preferred Skills: {top_job['Preferred Skills']}")
    print(f"Experience Req:   {top_job['Experience Requirement']} years")
    print(f"Education Req:    {top_job['Education Requirement']}")
    
    print("\n=============================================")
    print("FEATURE VALUES")
    print("=============================================")
    feats = extract_features(candidate, top_job)
    for k, v in feats.items():
        print(f"{k:<20}: {v:.4f}")
        
    print("\n=============================================")
    print("MATCH RESULTS")
    print("=============================================")
    print(f"Ranking Position:  {top_result['ranking_position']} of {len(sample_jobs)}")
    print(f"Model Probability: {top_result['model_probability']}%")
    print(f"Baseline Score:    {top_result['baseline_score']}%")
    print(f"Final Score:       {top_result['final_score']}%")
    
    print("\n=============================================")
    print("MATCH EXPLANATION")
    print("=============================================")
    print("Matched Skills:")
    for s in top_result['explanation']['matched_skills']:
        print(f"  - {s.title()}")
    print("Missing Skills:")
    for s in top_result['explanation']['missing_skills']:
        print(f"  - {s.title()}")
    print("\nWhy:")
    print(top_result['explanation']['why'])
    print("\nWalkthrough complete.")

if __name__ == '__main__':
    main()
