from typing import Dict, List

def generate_match_vector(student_skills: Dict[str, int], job_requirements: Dict[str, int]) -> List[int]:
    """
    Generates a binary match vector comparing student skills to job thresholds.
    
    Args:
        student_skills: Dictionary of skill names to student scores (e.g., {"Python": 85})
        job_requirements: Dictionary of skill names to job requirements (e.g., {"Python": 80})
        
    Returns:
        List of 1s (threshold met/exceeded) and 0s (threshold not met)
    """
    vector = []
    # Ensure consistent ordering by iterating through job_requirements
    for skill, threshold in job_requirements.items():
        # Clean suffix from job requirements if present (e.g., 'python_threshold' -> 'python')
        skill_clean = skill.replace("_threshold", "")
        
        student_score = student_skills.get(skill_clean, 0)
        
        if student_score >= threshold:
            vector.append(1)
        else:
            vector.append(0)
            
    return vector
