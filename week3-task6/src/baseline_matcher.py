def calculate_baseline_score(candidate_skills, job_required_skills):
    """
    Computes the baseline match score:
    match_score = (required_skills_matched / total_required_skills) * 100
    """
    if not job_required_skills:
        return 100.0
        
    # Handle cases where skills might be strings or None
    if not candidate_skills:
        candidate_skills = []
        
    candidate_skills_set = set([s.strip().lower() for s in candidate_skills if s and s.strip()])
    req_skills = [s.strip().lower() for s in job_required_skills if s and s.strip()]
    
    if not req_skills:
        return 100.0
        
    matched = [skill for skill in req_skills if skill in candidate_skills_set]
    
    score = (len(matched) / len(req_skills)) * 100
    return round(score, 2)
