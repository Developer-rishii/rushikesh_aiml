from typing import Dict, List

def generate_explanation(
    student_skills: Dict[str, int], 
    job_requirements: Dict[str, int],
    match_score: float,
    eligible: bool,
    match_vector: List[int]
) -> str:
    """
    Generates a human-readable explanation of the matching process.
    """
    total_skills = len(job_requirements)
    skills_met = sum(match_vector)
    
    explanation_lines = []
    explanation_lines.append(f"Candidate matched {skills_met} of {total_skills} required skills.\n")
    
    for skill, threshold in job_requirements.items():
        skill_clean = skill.replace("_threshold", "")
        if skill_clean.upper() == "SQL":
            readable_skill = "SQL"
        else:
            readable_skill = skill_clean.replace('_', ' ').title()
            
        student_score = student_skills.get(skill_clean, 0)
        
        if student_score >= threshold:
            mark = "[PASS]"
            operator = ">="
        else:
            mark = "[FAIL]"
            operator = "<"
            
        explanation_lines.append(f"{readable_skill}:")
        explanation_lines.append(f"{student_score} {operator} {threshold} {mark}\n")
        
    explanation_lines.append(f"Final Match Score: {match_score}%")
    explanation_lines.append(f"Eligible: {'Yes' if eligible else 'No'}")
    
    return "\n".join(explanation_lines)
