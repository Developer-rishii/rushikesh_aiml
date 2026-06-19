import pandas as pd
from typing import List, Dict, Any
from .matcher import calculate_match
from .explainability import generate_reasons

def rank_candidates(job: Dict[str, Any], students_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Evaluates all students for a given job and returns the Top-N ranked candidates.
    """
    candidates = []
    
    # Iterate through all students
    for _, student_row in students_df.iterrows():
        student_dict = student_row.to_dict()
        
        # Calculate match score and details
        score, details = calculate_match(student_dict, job)
        
        # Generate reasons
        reasons = generate_reasons(score, details)
        
        candidates.append({
            'student_id': student_dict.get('Student ID'),
            'student_name': student_dict.get('Name'),
            'match_score': score,
            'reasons': reasons
        })
        
    # Sort candidates by match_score descending
    ranked_candidates = sorted(candidates, key=lambda x: x['match_score'], reverse=True)
    
    return ranked_candidates[:top_n]
