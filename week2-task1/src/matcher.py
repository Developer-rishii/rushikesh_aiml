from typing import Dict, Any, Tuple

def calculate_match(student: Dict[str, Any], job: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Calculates the match score between a student and a job.
    Returns:
        Tuple containing:
        - match_score: integer from 0 to 100
        - details: dictionary containing score breakdown and raw values for explainability
    """
    score_details = {}
    total_score = 0.0
    
    # Weights for different components
    WEIGHTS = {
        'skills': 0.50,
        'cgpa': 0.20,
        'experience': 0.20,
        'soft_skills': 0.10
    }
    
    # 1. Technical Skills Match (50 points max)
    skill_reqs = job.get('Skill Requirements', {})
    student_skills = student.get('Skills', {})
    
    skill_score = 0.0
    skill_details = {}
    
    if not skill_reqs:
        # If no specific skills required, give full points for this section
        skill_score = 100.0
    else:
        for skill, min_score in skill_reqs.items():
            student_score = student_skills.get(skill, 0)
            diff = student_score - min_score
            
            if student_score >= min_score:
                # Meets or exceeds requirement
                # Base 100% for this skill, plus up to 20% bonus for exceeding
                bonus = min(20, diff) # Cap bonus
                item_score = 100 + bonus
            else:
                # Falls short
                # Penalty proportional to how much they fell short
                penalty = abs(diff) * 2
                item_score = max(0, 100 - penalty)
                
            skill_details[skill] = {
                'student_score': student_score,
                'required': min_score,
                'diff': diff,
                'points_awarded': item_score
            }
            skill_score += item_score
            
        # Average skill score capped at 100
        skill_score = min(100.0, skill_score / len(skill_reqs))
        
    score_details['skills'] = {
        'total': skill_score,
        'breakdown': skill_details
    }
    total_score += skill_score * WEIGHTS['skills']
    
    # 2. CGPA Match (20 points max)
    min_cgpa = job.get('Minimum CGPA', 0.0)
    student_cgpa = student.get('CGPA', 0.0)
    
    cgpa_diff = student_cgpa - min_cgpa
    if student_cgpa >= min_cgpa:
        cgpa_score = min(100.0, 100.0 + (cgpa_diff * 10)) # Small bonus for higher CGPA
    else:
        cgpa_score = max(0.0, 100.0 - (abs(cgpa_diff) * 30)) # Harsh penalty for lower CGPA
        
    score_details['cgpa'] = {
        'total': cgpa_score,
        'student_cgpa': student_cgpa,
        'required': min_cgpa,
        'diff': cgpa_diff
    }
    total_score += cgpa_score * WEIGHTS['cgpa']
    
    # 3. Experience Match (20 points max)
    req_exp = job.get('Experience Requirement', 0)
    # We equate 1 internship or 2 projects to 1 "unit" of experience
    projects = student.get('Project Count', 0)
    internships = student.get('Internship Count', 0)
    
    student_exp_units = internships + (projects * 0.5)
    exp_diff = student_exp_units - req_exp
    
    if student_exp_units >= req_exp:
        exp_score = min(100.0, 100.0 + (exp_diff * 10))
    else:
        exp_score = max(0.0, 100.0 - (abs(exp_diff) * 40))
        
    score_details['experience'] = {
        'total': exp_score,
        'student_units': student_exp_units,
        'required': req_exp,
        'internships': internships,
        'projects': projects
    }
    total_score += exp_score * WEIGHTS['experience']
    
    # 4. Soft Skills Match (10 points max)
    comm_score = student.get('Communication Score', 0)
    apt_score = student.get('Aptitude Score', 0)
    
    # Assume 75 is a baseline expected soft skill score
    soft_skills_avg = (comm_score + apt_score) / 2
    if soft_skills_avg >= 75:
        soft_skills_score = min(100.0, 100.0 + (soft_skills_avg - 75))
    else:
        soft_skills_score = max(0.0, 100.0 - ((75 - soft_skills_avg) * 2))
        
    score_details['soft_skills'] = {
        'total': soft_skills_score,
        'comm': comm_score,
        'apt': apt_score
    }
    total_score += soft_skills_score * WEIGHTS['soft_skills']
    
    # Final Score
    final_score_rounded = int(round(total_score))
    
    return final_score_rounded, score_details
