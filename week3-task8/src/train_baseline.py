import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.linear_model import LogisticRegression

def regenerate_data():
    os.makedirs('d:/Placemux-aiml/week3-task8/data', exist_ok=True)
    os.makedirs('d:/Placemux-aiml/week3-task8/models', exist_ok=True)

    skills_pool = ['Python', 'SQL', 'React', 'Node.js', 'AWS', 'Docker', 'Machine Learning', 'Java', 'C++', 'Git']
    edu_levels = ['Bachelors', 'Masters', 'PhD']
    certs_pool = ['AWS Certified', 'Google Cloud', 'Scrum Master', 'PMP', 'Cisco', 'CompTIA']

    # Generate Candidate Profiles
    candidates = []
    for i in range(1, 301):
        c_skills = random.sample(skills_pool, random.randint(2, 6))
        c_certs = random.sample(certs_pool, random.randint(0, 3))
        candidates.append({
            'candidate_id': i,
            'name': f'Candidate_{i}',
            'skills': ','.join(c_skills),
            'experience_years': round(random.uniform(0, 10), 1),
            'education': random.choice(edu_levels),
            'certifications': ','.join(c_certs)
        })
    df_c = pd.DataFrame(candidates)
    df_c.to_csv('d:/Placemux-aiml/week3-task8/data/candidate_profiles.csv', index=False)

    # Generate Jobs
    jobs = []
    for i in range(1, 121):
        j_skills = random.sample(skills_pool, random.randint(3, 7))
        j_certs = random.sample(certs_pool, random.randint(0, 2))
        jobs.append({
            'job_id': i,
            'title': f'Job_{i}',
            'required_skills': ','.join(j_skills),
            'min_experience': round(random.uniform(0, 5), 1),
            'required_education': random.choice(edu_levels),
            'required_certifications': ','.join(j_certs)
        })
    df_j = pd.DataFrame(jobs)
    df_j.to_csv('d:/Placemux-aiml/week3-task8/data/jobs.csv', index=False)
    
    return df_c, df_j

def extract_features(c, j):
    c_skills = set(c['skills'].split(',')) if c['skills'] else set()
    j_skills = set(j['required_skills'].split(',')) if j['required_skills'] else set()
    skill_overlap = len(c_skills.intersection(j_skills)) / max(len(j_skills), 1)

    exp_gap = c['experience_years'] - j['min_experience']
    
    # Simple education match: 1 if match or higher is simulated (exact match for simplicity)
    edu_match = 1 if c['education'] == j['required_education'] else 0
    
    c_certs = set(c['certifications'].split(',')) if c['certifications'] else set()
    j_certs = set(j['required_certifications'].split(',')) if j['required_certifications'] else set()
    cert_match_count = len(c_certs.intersection(j_certs))
    
    return [skill_overlap, exp_gap, edu_match, cert_match_count]

def build_model_and_history(df_c, df_j):
    records = []
    for _ in range(2000):
        c = df_c.sample(1).iloc[0]
        j = df_j.sample(1).iloc[0]
        
        feats = extract_features(c, j)
        skill_overlap, exp_gap, edu_match, cert_match_count = feats
        
        # True success based on strong combination of features
        # Must have good overlap, not too much negative exp gap
        success_prob = 0.1
        if skill_overlap >= 0.5 and exp_gap >= -1.0:
            success_prob = 0.8
        if skill_overlap >= 0.8 and edu_match == 1:
            success_prob = 0.95
        if skill_overlap < 0.3:
            success_prob = 0.05
            
        is_success = 1 if random.random() < success_prob else 0
        
        records.append({
            'candidate_id': c['candidate_id'],
            'job_id': j['job_id'],
            'skill_overlap': skill_overlap,
            'exp_gap': exp_gap,
            'edu_match': edu_match,
            'cert_match_count': cert_match_count,
            'is_success': is_success
        })
        
    df_train = pd.DataFrame(records)
    
    X = df_train[['skill_overlap', 'exp_gap', 'edu_match', 'cert_match_count']]
    y = df_train['is_success']
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    
    # Save model
    with open('d:/Placemux-aiml/week3-task8/models/baseline_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    # Generate prediction scores for history
    probs = clf.predict_proba(X)[:, 1]
    df_train['prediction_score'] = np.round(probs * 100, 2)
    df_train['baseline_score'] = np.round(df_train['skill_overlap'] * 100, 2)
    
    df_train.to_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv', index=False)
    
    print("Model trained and saved. Real match history generated.")
    
    # Print 5 real pairs
    print("\n--- 5 Real Candidate-Job Pairs ---")
    for idx, row in df_train.head(5).iterrows():
        print(f"Cand {int(row['candidate_id'])} & Job {int(row['job_id'])}:")
        print(f"  Baseline Score: {row['baseline_score']}% (Skill overlap)")
        print(f"  Prediction Score: {row['prediction_score']}% (LogisticRegression)")
        print(f"  Is Success: {int(row['is_success'])}\n")

if __name__ == '__main__':
    df_c, df_j = regenerate_data()
    build_model_and_history(df_c, df_j)
