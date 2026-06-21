import pandas as pd

def compute_features(df_students, df_jobs, df_applications=None):
    """
    Computes feature matrix for candidate-job pairs.
    If df_applications is None, computes cross product or expects pairs to be formed.
    For this use case, we assume we merge students and jobs on the applications dataframe, 
    or if we just want pairs, we pass a dataframe of pairs.
    """
    
    # If df_applications is provided, merge students and jobs onto it
    if df_students is not None and df_jobs is not None:
        if df_applications is not None:
            df = df_applications.merge(df_students, on='student_id', how='left')
            df = df.merge(df_jobs, on='job_id', how='left')
        else:
            raise ValueError("Please provide df_applications")
    else:
        # Assume the third argument is actually the pre-merged dataframe
        df = df_applications.copy()

    # 1. Skill Overlap Score: matched_skills / required_skills
    def calc_skill_overlap(row):
        s_skills = set(row['skills'].split(", ")) if pd.notna(row['skills']) else set()
        j_skills = set(row['required_skills'].split(", ")) if pd.notna(row['required_skills']) else set()
        if len(j_skills) == 0:
            return 1.0
        return len(s_skills.intersection(j_skills)) / len(j_skills)
        
    df['skill_overlap'] = df.apply(calc_skill_overlap, axis=1)
    
    # 2. Experience Match
    df['experience_match'] = (df['experience_years'] >= df['min_experience']).astype(int)
    
    # 3. Education Match
    edu_score = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    
    def calc_edu_match(row):
        s_edu = row['education']
        j_edu = row['education_required']
        if pd.isna(s_edu) or pd.isna(j_edu):
            return 0
        return 1 if edu_score.get(s_edu, 0) >= edu_score.get(j_edu, 0) else 0

    df['education_match'] = df.apply(calc_edu_match, axis=1)
    
    # 4. Location Match
    def calc_loc_match(row):
        return 1 if row['location_x'] == row['location_y'] or row['location_y'] == 'Remote' else 0
        
    df['location_match'] = df.apply(calc_loc_match, axis=1)
    
    features = ['skill_overlap', 'experience_match', 'education_match', 'location_match']
    if 'successful_match' in df.columns:
        features.append('successful_match')
        
    # Return the feature dataframe and also keep IDs for tracking
    cols_to_keep = ['student_id', 'job_id'] + features
    return df[cols_to_keep]

if __name__ == "__main__":
    df_students = pd.read_csv("data/students.csv")
    df_jobs = pd.read_csv("data/jobs.csv")
    df_apps = pd.read_csv("data/applications.csv")
    
    df_features = compute_features(df_students, df_jobs, df_apps)
    df_features.to_csv("data/features.csv", index=False)
    print("Features engineered and saved to data/features.csv")
