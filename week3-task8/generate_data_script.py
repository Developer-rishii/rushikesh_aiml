import pandas as pd
import numpy as np
import random
import os
import pickle

os.makedirs('d:/Placemux-aiml/week3-task8/data', exist_ok=True)
os.makedirs('d:/Placemux-aiml/week3-task8/models', exist_ok=True)

# Generate Candidate Profiles
skills_pool = ['Python', 'SQL', 'React', 'Node.js', 'AWS', 'Docker', 'Machine Learning', 'Java', 'C++', 'Git']
candidates = []
for i in range(1, 301):
    c_skills = random.sample(skills_pool, random.randint(2, 6))
    candidates.append({
        'candidate_id': i,
        'name': f'Candidate_{i}',
        'skills': ','.join(c_skills),
        'experience_years': round(random.uniform(0, 10), 1)
    })
df_candidates = pd.DataFrame(candidates)
df_candidates.to_csv('d:/Placemux-aiml/week3-task8/data/candidate_profiles.csv', index=False)

# Generate Jobs
jobs = []
for i in range(1, 121):
    j_skills = random.sample(skills_pool, random.randint(3, 7))
    jobs.append({
        'job_id': i,
        'title': f'Job_{i}',
        'required_skills': ','.join(j_skills),
        'min_experience': round(random.uniform(0, 5), 1)
    })
df_jobs = pd.DataFrame(jobs)
df_jobs.to_csv('d:/Placemux-aiml/week3-task8/data/jobs.csv', index=False)

# Generate Match History (500 records)
matches = []
for i in range(600):
    c = random.choice(candidates)
    j = random.choice(jobs)
    
    # Calculate overlap
    c_set = set(c['skills'].split(','))
    j_set = set(j['required_skills'].split(','))
    overlap = len(c_set.intersection(j_set)) / len(j_set)
    
    baseline_score = overlap * 100
    prediction_score = min(100, max(0, baseline_score + random.uniform(-15, 15)))
    
    # Successful outcome logic (mostly depends on overlap but with noise)
    prob_success = overlap
    if prediction_score > 70: prob_success += 0.2
    
    is_success = 1 if random.random() < prob_success else 0
    
    matches.append({
        'candidate_id': c['candidate_id'],
        'job_id': j['job_id'],
        'baseline_score': round(baseline_score, 2),
        'prediction_score': round(prediction_score, 2),
        'is_success': is_success
    })
df_matches = pd.DataFrame(matches)
df_matches.to_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv', index=False)

# Dummy baseline model
with open('d:/Placemux-aiml/week3-task8/models/baseline_model.pkl', 'wb') as f:
    pickle.dump('dummy_model_from_task6', f)

print('Data generation complete.')
