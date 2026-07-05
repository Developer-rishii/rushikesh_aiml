"""
generate_data.py

Step 1: Generate realistic-scale synthetic data.
Outputs to data/:
- students.csv
- student_skills.csv
- jobs.csv
- job_skills.csv
- outcomes.csv
"""

import os
import random
import pandas as pd
import numpy as np

# Ensure data dir exists
os.makedirs("data", exist_ok=True)

# Configuration
NUM_COLLEGES = 20
NUM_STUDENTS = 5000
NUM_COMPANIES = 80
NUM_JOBS = 500
NUM_SKILLS = 150

np.random.seed(42)
random.seed(42)

def generate_data():
    print("=== Generating Synthetic Data ===")
    
    # 1. Taxonomy
    skills_taxonomy = [f"skill_{i}" for i in range(NUM_SKILLS)]
    colleges = [f"college_{i}" for i in range(NUM_COLLEGES)]
    companies = [f"company_{i}" for i in range(NUM_COMPANIES)]
    
    # 2. Students
    print(f"Generating {NUM_STUDENTS} students across {NUM_COLLEGES} colleges...")
    student_records = []
    for i in range(NUM_STUDENTS):
        student_records.append({
            "student_id": f"student_{i}",
            "college_id": random.choice(colleges),
            "years_of_experience": max(0, int(np.random.normal(2, 1.5))),
            "degree": random.choice(["Bachelors", "Masters", "PhD", "None"])
        })
    students_df = pd.DataFrame(student_records)
    students_df.to_csv("data/students.csv", index=False)
    
    # 3. Student Skills (10-30 skills per student)
    print("Generating student skills...")
    student_skills_records = []
    
    # Deliberate edge cases
    cold_start_students = set(random.sample(range(NUM_STUDENTS), 50)) # 50 students with zero skills
    duplicate_skill_students = set(random.sample(range(NUM_STUDENTS), 100)) # 100 students have duplicates
    
    for i in range(NUM_STUDENTS):
        if i in cold_start_students:
            continue
            
        num_skills = random.randint(10, 30)
        student_skills = random.sample(skills_taxonomy, num_skills)
        
        for skill in student_skills:
            prof = random.randint(10, 100)
            # Inject 3-5% missing proficiency scores
            if random.random() < 0.04:
                prof = np.nan
                
            student_skills_records.append({
                "student_id": f"student_{i}",
                "skill_id": skill,
                "proficiency": prof
            })
            
            # Inject duplicates
            if i in duplicate_skill_students and random.random() < 0.1:
                student_skills_records.append({
                    "student_id": f"student_{i}",
                    "skill_id": skill,
                    "proficiency": prof - 5 if pd.notnull(prof) else np.nan # Conflicting duplicate
                })
                
    student_skills_df = pd.DataFrame(student_skills_records)
    student_skills_df.to_csv("data/student_skills.csv", index=False)
    
    # 4. Jobs
    print(f"Generating {NUM_JOBS} jobs across {NUM_COMPANIES} companies...")
    job_records = []
    for i in range(NUM_JOBS):
        job_records.append({
            "job_id": f"job_{i}",
            "company_id": random.choice(companies),
            "seniority_level": random.randint(0, 5) # 0=Intern, 5=Staff
        })
    jobs_df = pd.DataFrame(job_records)
    jobs_df.to_csv("data/jobs.csv", index=False)
    
    # 5. Job Requirements (5-15 required skills per job)
    print("Generating job requirements...")
    job_skills_records = []
    
    zero_req_jobs = set(random.sample(range(NUM_JOBS), 10)) # 10 jobs with zero requirements
    
    for i in range(NUM_JOBS):
        if i in zero_req_jobs:
            continue
            
        num_reqs = random.randint(5, 15)
        req_skills = random.sample(skills_taxonomy, num_reqs)
        
        for skill in req_skills:
            min_prof = random.randint(30, 80)
            job_skills_records.append({
                "job_id": f"job_{i}",
                "skill_id": skill,
                "min_proficiency": min_prof
            })
    job_skills_df = pd.DataFrame(job_skills_records)
    job_skills_df.to_csv("data/job_skills.csv", index=False)
    
    # 6. Outcomes (Training Signal)
    # We will simulate 15 applications per student
    print("Simulating historical outcomes (was_shortlisted, was_hired)...")
    
    # Pre-compute maps for fast lookup
    student_exp_map = students_df.set_index("student_id")["years_of_experience"].to_dict()
    job_sen_map = jobs_df.set_index("job_id")["seniority_level"].to_dict()
    
    # Group skills
    student_skills_grouped = student_skills_df.groupby("student_id").apply(
        lambda x: dict(zip(x["skill_id"], x["proficiency"]))
    ).to_dict()
    
    job_skills_grouped = job_skills_df.groupby("job_id").apply(
        lambda x: dict(zip(x["skill_id"], x["min_proficiency"]))
    ).to_dict()
    
    outcome_records = []
    
    for i in range(NUM_STUDENTS):
        sid = f"student_{i}"
        s_exp = student_exp_map[sid]
        s_skills = student_skills_grouped.get(sid, {})
        
        # Randomly apply to 15 jobs
        applied_jobs = random.sample(range(NUM_JOBS), 15)
        
        for j in applied_jobs:
            jid = f"job_{j}"
            j_sen = job_sen_map[jid]
            j_skills = job_skills_grouped.get(jid, {})
            
            # Feature 1: Skill overlap
            if len(j_skills) == 0:
                overlap_ratio = 1.0 # If no reqs, perfect overlap
            else:
                matched_skills = 0
                for req_s, min_p in j_skills.items():
                    if req_s in s_skills:
                        prof = s_skills[req_s]
                        if pd.notnull(prof) and prof >= min_p:
                            matched_skills += 1
                overlap_ratio = matched_skills / len(j_skills)
                
            # Feature 2: Experience fit
            exp_diff = s_exp - j_sen
            exp_fit_score = 1.0 / (1.0 + abs(exp_diff)) # 1.0 is perfect fit
            
            # Add noise and compute ground truth likelihood
            # Misleading feature: random noise (e.g., resume formatting, interview luck)
            noise = np.random.normal(0, 0.15)
            
            p_shortlist_raw = (0.6 * overlap_ratio) + (0.3 * exp_fit_score) + noise
            was_shortlisted = int(p_shortlist_raw > 0.5)
            
            p_hire_raw = p_shortlist_raw + np.random.normal(0, 0.1)
            was_hired = int(was_shortlisted and (p_hire_raw > 0.65))
            
            outcome_records.append({
                "student_id": sid,
                "job_id": jid,
                "was_shortlisted": was_shortlisted,
                "was_hired": was_hired
            })
            
    outcomes_df = pd.DataFrame(outcome_records)
    outcomes_df.to_csv("data/outcomes.csv", index=False)
    
    # 7. Print Stats
    print("\n=== Dataset Generation Complete ===")
    print(f"Students generated: {len(students_df)}")
    print(f"Jobs generated: {len(jobs_df)}")
    print(f"Student Skills entries: {len(student_skills_df)} (Missing prof: {student_skills_df['proficiency'].isna().sum()})")
    print(f"Job Skills entries: {len(job_skills_df)}")
    print(f"Outcomes (applications): {len(outcomes_df)}")
    print(f"  - Shortlisted: {outcomes_df['was_shortlisted'].sum()} ({outcomes_df['was_shortlisted'].mean():.1%})")
    print(f"  - Hired: {outcomes_df['was_hired'].sum()} ({outcomes_df['was_hired'].mean():.1%})")
    print("Edge cases injected: cold-start students, zero-req jobs, missing values, duplicates.")

if __name__ == "__main__":
    generate_data()
