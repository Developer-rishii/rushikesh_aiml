import os
import pandas as pd
import random

# Generate realistic skill pool
SKILL_POOL = ["Python", "Java", "SQL", "Machine Learning", "Data Analysis", "Docker", "AWS", "React", "Node.js", "Git", "C++", "C#", "Kubernetes", "Azure", "GCP", "Tableau", "PowerBI"]

def generate_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    candidates_path = os.path.join(data_dir, "candidate_profiles.csv")
    jobs_path = os.path.join(data_dir, "jobs.csv")
    
    random.seed(42) # Reproducible data generation
    
    # Generate Jobs (120+)
    jobs_data = []
    titles = ["Software Engineer", "Data Scientist", "Backend Developer", "Frontend Developer", "DevOps Engineer", "Machine Learning Engineer", "Data Analyst", "Full Stack Developer"]
    educations = ["Bachelor's", "Master's", "PhD", "None"]
    
    for i in range(150):
        title = random.choice(titles)
        num_skills = random.randint(2, 6)
        if random.random() < 0.05: # messy row: long skill list
            num_skills = random.randint(8, 12)
        required_skills = random.sample(SKILL_POOL, num_skills)
        min_experience = random.randint(0, 7)
        pref_education = random.choices(educations, weights=[60, 30, 5, 5])[0]
        jobs_data.append({
            "job_id": 1000 + i,
            "title": title,
            "required_skills": ",".join(required_skills),
            "minimum_experience": min_experience,
            "preferred_education": pref_education
        })
    pd.DataFrame(jobs_data).to_csv(jobs_path, index=False)
    
    # Generate Candidates (300+)
    candidates_data = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
    
    for i in range(400):
        name = random.choice(names) + " " + str(i)
        num_skills = random.randint(3, 8)
        if random.random() < 0.05: # messy row: only 1-2 skills
            num_skills = random.randint(1, 2)
        skills = random.sample(SKILL_POOL, num_skills)
        
        # some empty/null skills
        if random.random() < 0.02:
            skills = []
            
        experience = random.randint(0, 10)
        # some missing years_experience
        if random.random() < 0.02:
            experience = None
            
        education = random.choices(educations, weights=[60, 30, 5, 5])[0]
        certs = random.randint(0, 3)
        
        candidates_data.append({
            "candidate_id": 10000 + i,
            "name": name,
            "skills": ",".join(skills) if skills else "",
            "years_experience": experience,
            "education": education,
            "certifications": certs
        })
    pd.DataFrame(candidates_data).to_csv(candidates_path, index=False)

def load_data(data_dir="data"):
    candidates_path = os.path.join(data_dir, "candidate_profiles.csv")
    jobs_path = os.path.join(data_dir, "jobs.csv")
    
    if not os.path.exists(candidates_path) or not os.path.exists(jobs_path):
        print(f"Data files not found in {data_dir}. Generating...")
        generate_data(data_dir)
        
    candidates_df = pd.read_csv(candidates_path)
    jobs_df = pd.read_csv(jobs_path)
    return candidates_df, jobs_df

if __name__ == "__main__":
    # When run directly from src/, put data in ../data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    generate_data(data_dir)
    print("Data generated successfully.")
