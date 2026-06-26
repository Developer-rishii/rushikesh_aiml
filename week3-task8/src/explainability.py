def generate_reasons(match_data: dict, threshold: float) -> list:
    """
    Generate plain-English reasons for the guardrail decision.
    """
    reasons = []
    
    score = match_data.get('prediction_score', 0)
    if score < threshold:
        reasons.append(f"Match score {score:.1f}% is below the {threshold:.1f}% spend-safety threshold.")
    else:
        reasons.append(f"Match score {score:.1f}% passes the {threshold:.1f}% spend-safety threshold.")
        
    c_skills = set(match_data.get('candidate_skills', []))
    j_skills = set(match_data.get('job_skills', []))
    
    missing_skills = j_skills - c_skills
    if missing_skills:
        reasons.append(f"Missing required skills: {', '.join(missing_skills)}")
        
    if not c_skills.intersection(j_skills):
        reasons.append("Zero skill overlap with the job requirements.")
        
    c_exp = match_data.get('candidate_exp', 0)
    j_exp = match_data.get('job_exp', 0)
    if c_exp < j_exp:
        reasons.append(f"Experience gap: you have {c_exp} years, but the job requires {j_exp} years.")
        
    return reasons
