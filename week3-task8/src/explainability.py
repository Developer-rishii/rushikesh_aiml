def generate_reasons(match_data: dict, threshold: float) -> list:
    """
    Generate plain-English reasons for the guardrail decision.
    Reasons are derived from real feature values, never hardcoded.
    """
    reasons = []
    
    score = match_data.get('prediction_score', 0)
    if score < threshold:
        reasons.append(f"Match score {score:.1f}% is below the {threshold:.1f}% spend-safety threshold.")
    else:
        reasons.append(f"Match score {score:.1f}% passes the {threshold:.1f}% spend-safety threshold.")
        
    c_skills = set(match_data.get('candidate_skills', []))
    j_skills = set(match_data.get('job_skills', []))
    
    matched = c_skills.intersection(j_skills)
    missing_skills = j_skills - c_skills
    
    if matched:
        reasons.append(f"Matched skills: {', '.join(sorted(matched))}")
    
    if missing_skills:
        reasons.append(f"Missing required skills: {', '.join(sorted(missing_skills))}")
        
    if not c_skills.intersection(j_skills):
        reasons.append("Zero skill overlap with the job requirements.")
        
    c_exp = match_data.get('candidate_exp', 0)
    j_exp = match_data.get('job_exp', 0)
    if c_exp < j_exp:
        gap = round(j_exp - c_exp, 1)
        reasons.append(f"Experience gap of {gap} years: you have {c_exp} years, job requires {j_exp} years.")
    
    edu_match = match_data.get('edu_match', None)
    if edu_match == 0:
        reasons.append("Education level does not match the job requirement.")
    
    cert_count = match_data.get('cert_match_count', None)
    if cert_count is not None and cert_count == 0:
        reasons.append("No matching certifications with the job requirements.")
        
    return reasons
