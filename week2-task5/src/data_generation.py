import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

SKILLS_POOL = [
    "Python", "SQL", "Machine Learning", "Deep Learning", "Statistics",
    "NLP", "Computer Vision", "MLOps", "Data Engineering", "R",
    "Java", "C++", "AWS", "GCP", "Docker", "Kubernetes", "React", "Node.js"
]

TITLES_POOL = [
    "Data Scientist", "Machine Learning Engineer", "Data Analyst",
    "Data Engineer", "Software Engineer", "AI Researcher", "MLOps Engineer"
]

EDUCATION_POOL = ["Bachelors", "Masters", "PhD", "Bootcamp", "Self-Taught"]

def generate_jobs(num_jobs=50):
    jobs = []
    for i in range(1, num_jobs + 1):
        num_req_skills = random.randint(2, 5)
        num_pref_skills = random.randint(0, 3)
        
        req_skills = random.sample(SKILLS_POOL, num_req_skills)
        pref_skills = random.sample([s for s in SKILLS_POOL if s not in req_skills], num_pref_skills)
        
        job = {
            "job_id": f"J{i:03d}",
            "title": random.choice(TITLES_POOL),
            "required_skills": ", ".join(req_skills),
            "preferred_skills": ", ".join(pref_skills),
            "minimum_skill_score": random.randint(50, 85),
            "experience_required": random.randint(0, 5)
        }
        jobs.append(job)
    return pd.DataFrame(jobs)

def generate_students(num_students=300):
    students = []
    for i in range(1, num_students + 1):
        num_skills = random.randint(1, 8)
        skills = random.sample(SKILLS_POOL, num_skills)
        
        # Skill scores out of 100
        scores = {skill: random.randint(40, 100) for skill in skills}
        
        student = {
            "student_id": f"S{i:03d}",
            "verified_skills": ", ".join(skills),
            "skill_scores": str(scores),
            "experience": random.randint(0, 7),
            "education": random.choice(EDUCATION_POOL)
        }
        students.append(student)
    return pd.DataFrame(students)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("Generating Jobs...")
    jobs_df = generate_jobs(100) # generate 100 jobs
    jobs_df.to_csv("data/jobs.csv", index=False)
    print(f"Generated {len(jobs_df)} jobs.")
    
    print("Generating Students...")
    students_df = generate_students(500) # generate 500 students
    students_df.to_csv("data/students.csv", index=False)
    print(f"Generated {len(students_df)} students.")
    
    print("Datasets saved to data/")
