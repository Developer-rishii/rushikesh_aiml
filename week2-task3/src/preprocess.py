import pandas as pd
import numpy as np
import random
import os

def generate_data(num_students=5000, num_jobs=1000, num_applications=15000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Pools for random generation
    skills_pool = [
        "Python", "SQL", "Docker", "AWS", "Machine Learning", "Java", "C++", 
        "React", "Node.js", "Kubernetes", "GCP", "Azure", "NLP", "Computer Vision",
        "Data Engineering", "TensorFlow", "PyTorch", "Go", "Ruby", "TypeScript"
    ]
    
    education_levels = ["High School", "Bachelor", "Master", "PhD"]
    locations = ["New York", "San Francisco", "London", "Remote", "Berlin", "Austin", "Seattle", "Toronto", "Bangalore", "Singapore"]
    
    titles = [
        "Software Engineer", "Data Scientist", "Backend Developer", "Frontend Developer",
        "Machine Learning Engineer", "DevOps Engineer", "Data Engineer", "Product Manager",
        "Research Scientist", "Full Stack Developer"
    ]

    # Generate Students
    students = []
    for i in range(num_students):
        num_skills = random.randint(2, 6)
        student_skills = random.sample(skills_pool, k=num_skills)
        students.append({
            "student_id": f"S{i+1:05d}",
            "name": f"Student_{i+1}",
            "skills": ", ".join(student_skills),
            "experience_years": round(np.random.gamma(2, 2)), # right skewed distribution of experience
            "education": random.choices(education_levels, weights=[0.1, 0.5, 0.3, 0.1])[0],
            "location": random.choice(locations)
        })
    df_students = pd.DataFrame(students)
    
    # Generate Jobs
    jobs = []
    for i in range(num_jobs):
        num_req_skills = random.randint(3, 7)
        req_skills = random.sample(skills_pool, k=num_req_skills)
        jobs.append({
            "job_id": f"J{i+1:05d}",
            "job_title": random.choice(titles),
            "required_skills": ", ".join(req_skills),
            "min_experience": random.randint(0, 10),
            "education_required": random.choices(education_levels, weights=[0.05, 0.55, 0.3, 0.1])[0],
            "location": random.choice(locations)
        })
    df_jobs = pd.DataFrame(jobs)
    
    # Generate Applications (Target successful_match based on some logic so model can learn)
    applications = []
    
    education_score = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    
    for i in range(num_applications):
        student = df_students.sample(1).iloc[0]
        job = df_jobs.sample(1).iloc[0]
        
        # Calculate matching to bias the successful_match
        s_skills = set(student['skills'].split(", "))
        j_skills = set(job['required_skills'].split(", "))
        
        overlap = len(s_skills.intersection(j_skills)) / len(j_skills)
        exp_match = 1 if student['experience_years'] >= job['min_experience'] else 0
        edu_match = 1 if education_score[student['education']] >= education_score[job['education_required']] else 0
        loc_match = 1 if student['location'] == job['location'] or job['location'] == 'Remote' else 0
        
        # Probability of success
        prob = 0.5 * overlap + 0.3 * exp_match + 0.1 * edu_match + 0.1 * loc_match
        
        # Add some noise
        success = 1 if random.random() < prob else 0
        
        applications.append({
            "student_id": student['student_id'],
            "job_id": job['job_id'],
            "successful_match": success
        })
    
    df_applications = pd.DataFrame(applications)
    
    # Ensure data dir exists
    os.makedirs("data", exist_ok=True)
    
    df_students.to_csv("data/students.csv", index=False)
    df_jobs.to_csv("data/jobs.csv", index=False)
    df_applications.to_csv("data/applications.csv", index=False)
    
    print(f"Generated {len(df_students)} students, {len(df_jobs)} jobs, and {len(df_applications)} applications.")

if __name__ == "__main__":
    generate_data()
