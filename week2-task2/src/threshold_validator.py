from typing import Dict, Any, List

def validate_thresholds(student_skills: Dict[str, int], job_requirements: Dict[str, int]) -> Dict[str, Any]:
    """
    Validates thresholds and determines eligibility.
    
    Args:
        student_skills: Dictionary of skill names to student scores
        job_requirements: Dictionary of skill names to job requirements
        
    Returns:
        Dictionary containing 'eligible' (bool) and 'missing_skills' (List[str])
    """
    missing_skills = []
    
    for skill, threshold in job_requirements.items():
        skill_clean = skill.replace("_threshold", "")
        student_score = student_skills.get(skill_clean, 0)
        
        if student_score < threshold:
            # Reformat to human-readable
            if skill_clean.upper() == "SQL":
                readable_skill = "SQL"
            else:
                readable_skill = skill_clean.replace('_', ' ').title()
            missing_skills.append(readable_skill)
            
    return {
        "eligible": len(missing_skills) == 0,
        "missing_skills": missing_skills
    }
