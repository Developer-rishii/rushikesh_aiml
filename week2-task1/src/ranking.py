import pandas as pd
from typing import List, Dict, Any
from .matcher import calculate_match
from .explainability import generate_reasons

def rank_candidates(job: Dict[str, Any], students_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Evaluates all students for a given job and returns the Top-N ranked candidates.
    """
    candidates = []
    seen_ids = set()
    
    # Iterate through all students
    for _, student_row in students_df.iterrows():
        student_dict = student_row.to_dict()
        student_id = student_dict.get('Student ID')
        
        if student_id in seen_ids:
            continue
        seen_ids.add(student_id)
        
        # Calculate match score and details
        score, details = calculate_match(student_dict, job)
        
        # Generate reasons
        reasons = generate_reasons(score, details)
        
        skills_score = details.get('skills', {}).get('total', 0.0)
        cgpa_score = details.get('cgpa', {}).get('total', 0.0)
        
        candidates.append({
            'student_id': student_id,
            'student_name': student_dict.get('Name'),
            'match_score': score,
            'skills_score': skills_score,
            'cgpa_score': cgpa_score,
            'reasons': reasons
        })
        
    # Sort candidates by match_score descending, then skills_score, then cgpa_score
    ranked_candidates = sorted(
        candidates, 
        key=lambda x: (x['match_score'], x['skills_score'], x['cgpa_score']), 
        reverse=True
    )
    
    # Inject tie-breaker reasons where applicable
    for i in range(len(ranked_candidates) - 1):
        curr = ranked_candidates[i]
        nxt = ranked_candidates[i+1]
        
        if curr['match_score'] == nxt['match_score']:
            if curr['skills_score'] > nxt['skills_score']:
                curr['reasons'].append("🔹 Ranked above tied candidates due to higher skills score.")
            elif curr['cgpa_score'] > nxt['cgpa_score']:
                curr['reasons'].append("🔹 Ranked above tied candidates due to higher CGPA score.")
                
    # Remove internal fields for output consistency
    for c in ranked_candidates:
        c.pop('skills_score', None)
        c.pop('cgpa_score', None)
        
    return ranked_candidates[:top_n]
