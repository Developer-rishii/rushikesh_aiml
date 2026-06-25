def get_baseline_score(candidate, job):
    """
    Computes the Skill Overlap Baseline score.
    score = (number_of_matching_skills / number_of_required_skills)
    """
    c_skills = set([s.strip().lower() for s in str(candidate.get('Skills', '')).split(',') if s.strip()])
    j_req_skills = set([s.strip().lower() for s in str(job.get('Required Skills', '')).split(',') if s.strip()])
    
    if not j_req_skills:
        return 1.0 # If no required skills, baseline is perfect
        
    overlap = len(c_skills.intersection(j_req_skills))
    return overlap / len(j_req_skills)
