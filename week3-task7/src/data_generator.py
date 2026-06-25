import random
import pandas as pd
import numpy as np
import os

# Configuration
N_CANDIDATES = 1000
N_JOBS = 200
N_PAIRS = 15000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../data')
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dictionaries
SKILLS_POOL = [
    'Python', 'SQL', 'Pandas', 'Machine Learning', 'Deep Learning',
    'Java', 'C++', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
    'React', 'Node.js', 'TypeScript', 'Power BI', 'Tableau', 'Excel',
    'Communication', 'Leadership', 'Agile', 'Scrum', 'Git', 'CI/CD'
]

EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']
CERTIFICATIONS_POOL = [
    'AWS Solutions Architect', 'Google Cloud Engineer', 'PMP',
    'Scrum Master', 'Data Science Professional', 'Machine Learning Specialization'
]

def generate_candidates(n):
    candidates = []
    for i in range(1, n + 1):
        num_skills = random.randint(2, 10)
        skills = random.sample(SKILLS_POOL, num_skills)
        exp = random.randint(0, 15)
        
        # Education distribution
        edu_rand = random.random()
        if edu_rand < 0.1: edu = 'High School'
        elif edu_rand < 0.6: edu = 'Bachelor'
        elif edu_rand < 0.9: edu = 'Master'
        else: edu = 'PhD'
        
        num_certs = random.randint(0, 3)
        certs = random.sample(CERTIFICATIONS_POOL, num_certs)
        
        projects = random.randint(0, 5)
        
        candidates.append({
            'Candidate ID': f'C_{i}',
            'Skills': ','.join(skills),
            'Experience Years': exp,
            'Education': edu,
            'Certifications': ','.join(certs),
            'Projects': projects
        })
    return pd.DataFrame(candidates)

def generate_jobs(n):
    jobs = []
    for i in range(1, n + 1):
        num_req_skills = random.randint(2, 6)
        num_pref_skills = random.randint(0, 3)
        
        all_sampled_skills = random.sample(SKILLS_POOL, num_req_skills + num_pref_skills)
        req_skills = all_sampled_skills[:num_req_skills]
        pref_skills = all_sampled_skills[num_req_skills:]
        
        exp_req = random.randint(0, 10)
        
        # Education requirement distribution
        edu_rand = random.random()
        if edu_rand < 0.2: edu_req = 'High School'
        elif edu_rand < 0.7: edu_req = 'Bachelor'
        elif edu_rand < 0.95: edu_req = 'Master'
        else: edu_req = 'PhD'
        
        jobs.append({
            'Job ID': f'J_{i}',
            'Required Skills': ','.join(req_skills),
            'Preferred Skills': ','.join(pref_skills),
            'Experience Requirement': exp_req,
            'Education Requirement': edu_req
        })
    return pd.DataFrame(jobs)

def _education_score(edu):
    return EDUCATION_LEVELS.index(edu)

def generate_pairs(candidates_df, jobs_df, n_pairs):
    pairs = []
    c_ids = candidates_df['Candidate ID'].tolist()
    j_ids = jobs_df['Job ID'].tolist()
    
    cand_dict = candidates_df.set_index('Candidate ID').to_dict('index')
    job_dict = jobs_df.set_index('Job ID').to_dict('index')
    
    # Generate random pairs
    sampled_pairs = set()
    while len(sampled_pairs) < n_pairs:
        c = random.choice(c_ids)
        j = random.choice(j_ids)
        sampled_pairs.add((c, j))
        
    for c, j in sampled_pairs:
        cand = cand_dict[c]
        job = job_dict[j]
        
        c_skills = set(cand['Skills'].split(',')) if cand['Skills'] else set()
        j_req_skills = set(job['Required Skills'].split(',')) if job['Required Skills'] else set()
        
        # Heuristic for is_match label
        # 1. Experience requirement met
        exp_met = cand['Experience Years'] >= job['Experience Requirement']
        
        # 2. Education requirement met
        edu_met = _education_score(cand['Education']) >= _education_score(job['Education Requirement'])
        
        # 3. Skill overlap
        if len(j_req_skills) > 0:
            skill_overlap = len(c_skills.intersection(j_req_skills)) / len(j_req_skills)
        else:
            skill_overlap = 1.0
            
        # Decision logic (realistic heuristic)
        if exp_met and edu_met and skill_overlap >= 0.5:
            is_match = 1
        elif exp_met and skill_overlap >= 0.8: # high skill overlap can compensate for education
            is_match = 1
        else:
            is_match = 0
            
        # Add 5% noise
        if random.random() < 0.05:
            is_match = 1 - is_match
            
        pairs.append({
            'Candidate ID': c,
            'Job ID': j,
            'is_match': is_match
        })
        
    return pd.DataFrame(pairs)

def main():
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {N_CANDIDATES} candidates...")
    candidates = generate_candidates(N_CANDIDATES)
    candidates.to_csv(os.path.join(OUTPUT_DIR, 'candidates.csv'), index=False)
    
    print(f"Generating {N_JOBS} jobs...")
    jobs = generate_jobs(N_JOBS)
    jobs.to_csv(os.path.join(OUTPUT_DIR, 'jobs.csv'), index=False)
    
    print(f"Generating {N_PAIRS} candidate-job pairs...")
    pairs = generate_pairs(candidates, jobs, N_PAIRS)
    
    # Shuffle pairs
    pairs = pairs.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Split 70/15/15
    train_end = int(0.7 * len(pairs))
    val_end = int(0.85 * len(pairs))
    
    train_pairs = pairs.iloc[:train_end]
    val_pairs = pairs.iloc[train_end:val_end]
    test_pairs = pairs.iloc[val_end:]
    
    train_pairs.to_csv(os.path.join(OUTPUT_DIR, 'train_pairs.csv'), index=False)
    val_pairs.to_csv(os.path.join(OUTPUT_DIR, 'val_pairs.csv'), index=False)
    test_pairs.to_csv(os.path.join(OUTPUT_DIR, 'test_pairs.csv'), index=False)
    
    print("Data generation complete!")
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")
    print("Positive class ratio (train):", train_pairs['is_match'].mean())

if __name__ == '__main__':
    main()
