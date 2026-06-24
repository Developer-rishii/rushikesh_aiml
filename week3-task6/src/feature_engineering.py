import pandas as pd
import numpy as np

def extract_features(candidates_df, jobs_df):
    """
    Cross join candidates and jobs to create a pairwise feature table.
    """
    candidates_df['_tmp'] = 1
    jobs_df['_tmp'] = 1
    pairs = pd.merge(candidates_df, jobs_df, on='_tmp').drop('_tmp', axis=1)
    
    # Remove _tmp from original dataframes
    candidates_df.drop('_tmp', axis=1, inplace=True)
    jobs_df.drop('_tmp', axis=1, inplace=True)
    
    features = []
    
    for _, row in pairs.iterrows():
        # Parse skills
        c_skills_raw = str(row.get('skills', ''))
        c_skills = [s.strip().lower() for s in c_skills_raw.split(',')] if c_skills_raw and c_skills_raw != 'nan' else []
        c_skills_set = set(c_skills)
        
        j_skills_raw = str(row.get('required_skills', ''))
        j_skills = [s.strip().lower() for s in j_skills_raw.split(',')] if j_skills_raw and j_skills_raw != 'nan' else []
        
        union_len = len(c_skills_set.union(j_skills))
        overlap_len = len(c_skills_set.intersection(j_skills))
        skill_overlap_percentage = (overlap_len / union_len * 100) if union_len > 0 else 0.0
        
        required_skill_coverage = (overlap_len / len(j_skills) * 100) if len(j_skills) > 0 else 100.0
        
        c_exp = row.get('years_experience')
        if pd.isna(c_exp):
            c_exp = 0
        j_exp = row.get('minimum_experience', 0)
        experience_gap = c_exp - j_exp
        
        c_edu = str(row.get('education', 'None'))
        j_edu = str(row.get('preferred_education', 'None'))
        education_map = {"None": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        education_match = 1 if education_map.get(c_edu, 0) >= education_map.get(j_edu, 0) else 0
        
        certification_match_count = int(row.get('certifications', 0) if not pd.isna(row.get('certifications')) else 0)
        
        # Generate Label
        label = 1 if (required_skill_coverage >= 60 and experience_gap >= -1) else 0
        
        features.append({
            'candidate_id': int(row['candidate_id']),
            'job_id': int(row['job_id']),
            'skill_overlap_percentage': round(skill_overlap_percentage, 2),
            'experience_gap': float(experience_gap),
            'education_match': int(education_match),
            'certification_match_count': certification_match_count,
            'required_skill_coverage': round(required_skill_coverage, 2),
            'label': label
        })
        
    return pd.DataFrame(features)

def extract_features_for_pair(candidate_dict, job_dict):
    """
    Extract features for a single candidate-job pair. Useful for inference.
    """
    c_df = pd.DataFrame([candidate_dict])
    j_df = pd.DataFrame([job_dict])
    return extract_features(c_df, j_df).iloc[0].to_dict()
