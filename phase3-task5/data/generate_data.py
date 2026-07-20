import pandas as pd
import numpy as np
import os
import random

def generate_data(num_candidates=1000, num_jobs=500, num_interactions=20000):
    np.random.seed(42)
    random.seed(42)

    # 1. Generate Candidates
    # Features: candidate_experience, candidate_skills_count, demographic_group
    candidate_ids = [f"C{i}" for i in range(num_candidates)]
    candidate_exp = np.random.randint(0, 15, num_candidates)
    candidate_skills = np.random.randint(1, 10, num_candidates)
    # Demographic group: 0 (Majority), 1 (Minority) for fairness evaluation
    demographic_group = np.random.choice([0, 1], p=[0.7, 0.3], size=num_candidates)
    
    candidates_df = pd.DataFrame({
        'candidate_id': candidate_ids,
        'candidate_exp': candidate_exp,
        'candidate_skills': candidate_skills,
        'demographic_group': demographic_group
    })

    # 2. Generate Jobs
    # Features: required_experience, required_skills_count, job_popularity
    job_ids = [f"J{i}" for i in range(num_jobs)]
    required_exp = np.random.randint(0, 10, num_jobs)
    required_skills = np.random.randint(1, 10, num_jobs)
    job_popularity = np.random.uniform(0.1, 1.0, num_jobs)
    
    jobs_df = pd.DataFrame({
        'job_id': job_ids,
        'required_exp': required_exp,
        'required_skills': required_skills,
        'job_popularity': job_popularity
    })

    # 3. Generate Interactions
    # Random candidate-job pairs
    selected_candidates = np.random.choice(candidate_ids, size=num_interactions)
    selected_jobs = np.random.choice(job_ids, size=num_interactions)
    
    interactions = pd.DataFrame({
        'candidate_id': selected_candidates,
        'job_id': selected_jobs
    })

    # Merge features
    interactions = interactions.merge(candidates_df, on='candidate_id', how='left')
    interactions = interactions.merge(jobs_df, on='job_id', how='left')

    # Calculate relevance
    # Relevance score increases if candidate exp >= required exp, and skills >= required
    exp_match = (interactions['candidate_exp'] >= interactions['required_exp']).astype(int)
    skill_match = (interactions['candidate_skills'] >= interactions['required_skills']).astype(int)
    
    # Introduce a slight negative bias against demographic_group 1 in real outcomes to simulate historical bias
    bias = interactions['demographic_group'] * -0.2
    
    interactions['raw_score'] = exp_match * 0.4 + skill_match * 0.4 + interactions['job_popularity'] * 0.2 + bias
    
    # Add some noise
    interactions['raw_score'] += np.random.normal(0, 0.1, num_interactions)
    
    # Convert raw score to labels: 0 (No Action), 1 (Click), 2 (Apply)
    def to_label(score):
        if score > 0.8:
            return 2 # Apply
        elif score > 0.5:
            return 1 # Click
        else:
            return 0 # Ignore
            
    interactions['relevance'] = interactions['raw_score'].apply(to_label)
    interactions.drop(columns=['raw_score'], inplace=True, errors='ignore')
    
    # Ensure no duplicates for same candidate-job pair
    interactions = interactions.groupby(['candidate_id', 'job_id']).first().reset_index()

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    interactions.to_csv('data/interactions.csv', index=False)
    candidates_df.to_csv('data/candidates.csv', index=False)
    jobs_df.to_csv('data/jobs.csv', index=False)
    print(f"Generated {len(interactions)} interactions.")

if __name__ == "__main__":
    generate_data()
