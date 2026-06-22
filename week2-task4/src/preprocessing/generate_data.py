import pandas as pd
import numpy as np
import random
import os

def generate_data(num_students=2000, num_jobs=500):
    np.random.seed(42)
    random.seed(42)

    all_skills = [
        "Python", "SQL", "Machine Learning", "Communication",
        "Data Analysis", "Java", "C++", "AWS", "Docker", "Kubernetes",
        "React", "Node.js", "Statistics", "Deep Learning", "NLP",
        "Project Management", "Agile", "Scrum", "Git"
    ]

    # Generate Students
    students = []
    for i in range(1, num_students + 1):
        num_skills = random.randint(2, 7)
        skills = random.sample(all_skills, num_skills)
        students.append({
            "student_id": 100 + i,
            "skills": ",".join(skills),
            "experience_years": random.randint(0, 10),
            "verified_score": random.randint(40, 100)
        })
    df_students = pd.DataFrame(students)

    # Generate Jobs
    jobs = []
    for i in range(1, num_jobs + 1):
        num_req_skills = random.randint(2, 5)
        req_skills = random.sample(all_skills, num_req_skills)
        jobs.append({
            "job_id": 500 + i,
            "required_skills": ",".join(req_skills),
            "minimum_score": random.randint(50, 90)
        })
    df_jobs = pd.DataFrame(jobs)

    # Generate Matches (Ranking Output)
    matches = []
    # Create ~3000 pairs
    for _ in range(3000):
        student = random.choice(students)
        job = random.choice(jobs)
        
        s_skills = set(student["skills"].split(","))
        j_skills = set(job["required_skills"].split(","))
        
        overlap = len(s_skills.intersection(j_skills))
        overlap_pct = (overlap / len(j_skills)) * 100 if len(j_skills) > 0 else 0
        
        # Calculate a fake "match_score" from ranking module
        # Combine overlap and verified score with some noise
        base_score = (overlap_pct * 0.6) + (student["verified_score"] * 0.3) + (random.randint(0, 10))
        match_score = min(max(int(base_score), 0), 100)
        
        # Determine actual label (matched = 1, not_matched = 0)
        # We assume a match is good if match_score > 70 and verified_score >= minimum_score
        is_matched = 1 if match_score > 65 and student["verified_score"] >= job["minimum_score"] - 5 else 0

        matches.append({
            "student_id": student["student_id"],
            "job_id": job["job_id"],
            "match_score": match_score,
            "matched": is_matched
        })
    df_matches = pd.DataFrame(matches)

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'data'), exist_ok=True)
    df_students.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'students.csv'), index=False)
    df_jobs.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'jobs.csv'), index=False)
    df_matches.to_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'matches.csv'), index=False)
    print("Generated data in data/ directory successfully!")

if __name__ == "__main__":
    generate_data()
