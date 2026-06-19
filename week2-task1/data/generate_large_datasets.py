import pandas as pd
import numpy as np
import random
import os

def generate_datasets(num_students=500, num_jobs=100):
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # 1. Generate Students
    students_data = []
    first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan', 'Fiona', 'George', 'Hannah', 'Ian', 'Jane', 'Kevin', 'Laura', 'Mike', 'Nina', 'Oscar', 'Paul', 'Quinn', 'Rachel', 'Sam', 'Tina', 'Uma', 'Victor', 'Wendy', 'Xavier', 'Yara', 'Zack']
    last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis']
    
    for i in range(1, num_students + 1):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        students_data.append({
            'Student ID': i,
            'Name': name,
            'Verified Python Score': np.random.randint(40, 100),
            'Verified SQL Score': np.random.randint(40, 100),
            'Verified ML Score': np.random.randint(40, 100),
            'Communication Score': np.random.randint(50, 100),
            'Aptitude Score': np.random.randint(50, 100),
            'Project Count': np.random.randint(0, 6),
            'Internship Count': np.random.randint(0, 4),
            'CGPA': round(np.random.uniform(5.0, 10.0), 1)
        })
        
    students_df = pd.DataFrame(students_data)
    
    # 2. Generate Jobs
    jobs_data = []
    companies = [f"TechCompany_{i}" for i in range(1, 21)] + [f"Startup_{i}" for i in range(1, 11)] + [f"Corp_{i}" for i in range(1, 11)]
    roles = ['Data Scientist', 'Data Analyst', 'ML Engineer', 'Python Developer', 'Data Engineer', 'Research Scientist', 'AI Specialist']
    all_skills = ['Python', 'SQL', 'ML']
    
    for i in range(1001, 1001 + num_jobs):
        role = random.choice(roles)
        company = random.choice(companies)
        
        # Determine skills based on role
        if role == 'Data Analyst':
            skills = ['Python', 'SQL'] if random.random() > 0.3 else ['SQL']
        elif role == 'Python Developer':
            skills = ['Python']
        elif role in ['ML Engineer', 'AI Specialist']:
            skills = ['Python', 'ML']
        else:
            # Data Scientist, Data Engineer, Research Scientist
            skills = random.sample(all_skills, k=random.randint(2, 3))
            
        req_skills_str = ",".join(skills)
        min_scores_str = ",".join([str(np.random.randint(60, 90)) for _ in skills])
        
        jobs_data.append({
            'Job ID': i,
            'Company Name': company,
            'Role': role,
            'Required Skills': req_skills_str,
            'Minimum Skill Scores': min_scores_str,
            'Minimum CGPA': round(np.random.uniform(6.5, 8.5), 1),
            'Experience Requirement': np.random.randint(0, 4)
        })
        
    jobs_df = pd.DataFrame(jobs_data)
    
    # Ensure data directory exists
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    students_df.to_csv(os.path.join(data_dir, 'students.csv'), index=False)
    jobs_df.to_csv(os.path.join(data_dir, 'jobs.csv'), index=False)
    
    print(f"Successfully generated {len(students_df)} students and {len(jobs_df)} jobs.")

if __name__ == "__main__":
    generate_datasets()
