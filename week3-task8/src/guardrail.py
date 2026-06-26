from .explainability import generate_reasons

def evaluate_guardrail(match_data: dict, threshold: float) -> dict:
    """
    Evaluate if a candidate-job match is fit to spend money on.
    """
    score = match_data.get('prediction_score', 0)
    
    c_skills = set(match_data.get('candidate_skills', []))
    j_skills = set(match_data.get('job_skills', []))
    overlap = len(c_skills.intersection(j_skills))
    
    status = "OK"
    if score < threshold or overlap == 0:
        status = "LOW_FIT_WARNING"
        
    reasons = generate_reasons(match_data, threshold)
    
    return {
        "fit_status": status,
        "match_score": score,
        "threshold": threshold,
        "reason": reasons
    }
