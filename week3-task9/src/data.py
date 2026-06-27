import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_skills():
    return ["Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "Node.js", "Java", "C++", "Go"]

def generate_data(num_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    all_skills = generate_skills()
    
    # Jobs
    job_skills_req = []
    job_seniority = []
    
    for _ in range(num_samples):
        # 2 to 5 skills required
        req = np.random.choice(all_skills, size=np.random.randint(2, 6), replace=False)
        job_skills_req.append(list(req))
        job_seniority.append(np.random.randint(1, 6)) # 1 to 5 levels
        
    # Students
    student_skills = []
    student_years_exp = []
    payment_status = []
    
    payment_options = ["paid", "failed", "pending", "refunded"]
    
    for _ in range(num_samples):
        # 1 to 6 skills
        has = np.random.choice(all_skills, size=np.random.randint(1, 7), replace=False)
        student_skills.append(list(has))
        student_years_exp.append(np.random.randint(0, 10))
        # Skew slightly towards paid
        payment_status.append(np.random.choice(payment_options, p=[0.6, 0.15, 0.15, 0.1]))

    df = pd.DataFrame({
        'job_skills': job_skills_req,
        'job_seniority': job_seniority,
        'student_skills': student_skills,
        'student_years_exp': student_years_exp,
        'payment_status': payment_status
    })

    # Ground truth logic:
    # A "good match" means overlap ratio is high, and years of experience is close or higher than seniority * 1.5
    
    overlap_ratios = []
    labels = []
    for i, row in df.iterrows():
        overlap = len(set(row['student_skills']).intersection(set(row['job_skills'])))
        total_req = len(row['job_skills'])
        ratio = overlap / total_req if total_req > 0 else 0
        overlap_ratios.append(ratio)
        
        # Ground truth formulation
        is_good = 1 if (ratio >= 0.6 and row['student_years_exp'] >= row['job_seniority']) else 0
        
        # Add some noise
        if np.random.rand() < 0.1:
            is_good = 1 - is_good
            
        labels.append(is_good)
        
    df['overlap_ratio'] = overlap_ratios
    df['is_good_match'] = labels
    
    # Features for the model:
    # We will expand these in the model layer, but data layer provides raw values.
    
    return df

def get_train_val_test_splits(df, random_state=42):
    # Train 60%, Val 20%, Test 20%
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)
    
    return train_df, val_df, test_df
