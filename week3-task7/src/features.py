import numpy as np

EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']

def _education_score(edu):
    if not edu or edu not in EDUCATION_LEVELS:
        return -1
    return EDUCATION_LEVELS.index(edu)

def extract_features(candidate, job):
    """
    Extracts features for a candidate-job pair.
    
    candidate: dict with keys: Skills, Experience Years, Education, Certifications, Projects
    job: dict with keys: Required Skills, Preferred Skills, Experience Requirement, Education Requirement
    """
    
    # Parse candidate
    c_skills = set([s.strip().lower() for s in str(candidate.get('Skills', '')).split(',') if s.strip()])
    try:
        c_exp = float(candidate.get('Experience Years', 0))
    except (ValueError, TypeError):
        c_exp = 0.0
        
    c_edu = candidate.get('Education', '')
    c_certs = [c.strip().lower() for c in str(candidate.get('Certifications', '')).split(',') if c.strip()]
    
    try:
        c_projects = float(candidate.get('Projects', 0))
    except (ValueError, TypeError):
        c_projects = 0.0
    
    # Parse job
    j_req_skills = set([s.strip().lower() for s in str(job.get('Required Skills', '')).split(',') if s.strip()])
    j_pref_skills = set([s.strip().lower() for s in str(job.get('Preferred Skills', '')).split(',') if s.strip()])
    j_all_skills = j_req_skills.union(j_pref_skills)
    
    try:
        j_exp_req = float(job.get('Experience Requirement', 0))
    except (ValueError, TypeError):
        j_exp_req = 0.0
    j_edu_req = job.get('Education Requirement', '')

    # 1. Skill Match Percentage
    if j_all_skills:
        skill_match_pct = len(c_skills.intersection(j_all_skills)) / len(j_all_skills)
    else:
        skill_match_pct = 1.0

    # 2. Required Skill Coverage
    if j_req_skills:
        req_skill_coverage = len(c_skills.intersection(j_req_skills)) / len(j_req_skills)
    else:
        req_skill_coverage = 1.0

    # 3. Preferred Skill Coverage
    if j_pref_skills:
        pref_skill_coverage = len(c_skills.intersection(j_pref_skills)) / len(j_pref_skills)
    else:
        pref_skill_coverage = 1.0

    # 4. Experience Match
    if j_exp_req > 0:
        exp_match = min(c_exp / j_exp_req, 1.0)
        # Bonus for extra experience? Let's just cap at 1.0 for simplicity, or 1.2 if we want to allow slight advantage.
        # Strict match: 1.0 if c_exp >= j_exp_req else fractional
    else:
        exp_match = 1.0

    # 5. Education Match
    c_edu_score = _education_score(c_edu)
    j_edu_score = _education_score(j_edu_req)
    
    if j_edu_score == -1: # No requirement or unknown
        edu_match = 1.0
    elif c_edu_score == -1: # Candidate has no valid education listed
        edu_match = 0.0
    else:
        edu_match = 1.0 if c_edu_score >= j_edu_score else float(c_edu_score) / j_edu_score if j_edu_score > 0 else 0.0

    # 6. Certification Match
    # Just checking if they have certs. Simple approach: min(num_certs / 2, 1.0) 
    # Because we don't have job cert requirements right now.
    cert_match = min(len(c_certs) / 2.0, 1.0)

    # 7. Project Relevance
    # Scale projects. Assume 3 projects is great (1.0).
    project_relevance = min(c_projects / 3.0, 1.0)

    return {
        'skill_match_pct': skill_match_pct,
        'req_skill_coverage': req_skill_coverage,
        'pref_skill_coverage': pref_skill_coverage,
        'exp_match': exp_match,
        'edu_match': edu_match,
        'cert_match': cert_match,
        'project_relevance': project_relevance
    }
