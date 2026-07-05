"""
explain.py

Generates a plain-English explanation for a prediction based on feature values.
"""

def generate_explanation(features_row):
    """
    Given a dictionary of features for a student-job match,
    returns a plain-English explanation of the score.
    """
    reasons = []
    
    # 1. Skill Overlap
    overlap = features_row.get("skill_overlap_ratio", 0)
    if overlap == 1.0:
        reasons.append("You possess all the required skills for this role.")
    elif overlap >= 0.7:
        reasons.append(f"You have a strong skill match ({overlap:.0%} of required skills).")
    elif overlap > 0.0:
        reasons.append(f"You have a partial skill match ({overlap:.0%} of required skills).")
    else:
        reasons.append("You do not have any of the required skills verified.")
        
    # 2. Proficiency Gap
    gap = features_row.get("proficiency_gap", 0)
    if gap <= 0 and overlap > 0:
        reasons.append("Your proficiency exceeds the minimum requirements.")
    elif gap > 0:
        reasons.append(f"There is a gap in your proficiency levels compared to what is expected.")
        
    # 3. Experience Fit
    exp = features_row.get("experience_fit", 0)
    if exp == 1.0:
        reasons.append("Your years of experience perfectly align with the seniority of this role.")
    elif exp >= 0.5:
        reasons.append("Your experience level is close to the expected seniority.")
    else:
        reasons.append("Your experience level differs significantly from the expected seniority.")
        
    return " ".join(reasons)
