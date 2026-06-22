import pandas as pd

def extract_features(student, job):
    """
    Extract features for a single student-job pair.
    student: dict-like with 'skills', 'verified_score', 'experience_years'
    job: dict-like with 'required_skills'
    """
    s_skills = set(student.get("skills", "").split(",")) if student.get("skills") else set()
    j_skills = set(job.get("required_skills", "").split(",")) if job.get("required_skills") else set()
    
    # Clean up empty strings
    s_skills = {s.strip() for s in s_skills if s.strip()}
    j_skills = {j.strip() for j in j_skills if j.strip()}

    matched_skills = s_skills.intersection(j_skills)
    missing_skills = j_skills - s_skills
    
    num_matching = len(matched_skills)
    num_missing = len(missing_skills)
    num_required = len(j_skills)
    
    overlap_pct = (num_matching / num_required) * 100 if num_required > 0 else 100.0

    return {
        "skill_overlap_percentage": overlap_pct,
        "verified_score": student.get("verified_score", 0),
        "experience_years": student.get("experience_years", 0),
        "number_of_matching_skills": num_matching,
        "number_of_missing_skills": num_missing,
        "matched_skills_list": list(matched_skills),
        "missing_skills_list": list(missing_skills)
    }

def build_feature_matrix(df_matches, df_students, df_jobs):
    """
    Build X and y for training/evaluation from raw dataframes.
    """
    df = df_matches.merge(df_students, on="student_id").merge(df_jobs, on="job_id")
    
    features_list = []
    y = []
    
    for _, row in df.iterrows():
        student_data = {
            "skills": row["skills"],
            "verified_score": row["verified_score"],
            "experience_years": row["experience_years"]
        }
        job_data = {
            "required_skills": row["required_skills"]
        }
        feats = extract_features(student_data, job_data)
        
        features_list.append({
            "skill_overlap_percentage": feats["skill_overlap_percentage"],
            "verified_score": feats["verified_score"],
            "experience_years": feats["experience_years"],
            "number_of_matching_skills": feats["number_of_matching_skills"],
            "number_of_missing_skills": feats["number_of_missing_skills"]
        })
        y.append(row["matched"])
        
    X = pd.DataFrame(features_list)
    return X, pd.Series(y)
