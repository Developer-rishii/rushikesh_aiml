from typing import List

def calculate_match_score(match_vector: List[int]) -> float:
    """
    Calculates the match score based on the binary match vector.
    
    Args:
        match_vector: List of 1s and 0s
        
    Returns:
        Percentage score (0.0 to 100.0) rounded to 2 decimal places.
    """
    if not match_vector:
        return 0.0
        
    total_required_skills = len(match_vector)
    skills_meeting_threshold = sum(match_vector)
    
    score = (skills_meeting_threshold / total_required_skills) * 100
    return round(score, 2)
