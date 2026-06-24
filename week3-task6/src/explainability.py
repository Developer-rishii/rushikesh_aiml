def generate_explanation(features_dict):
    """
    Generates a list of plain English explanation strings based on the features 
    of a candidate-job pair.
    """
    explanations = []
    
    # Skill overlap
    coverage = features_dict.get('required_skill_coverage', 0)
    if coverage == 100:
        explanations.append("All required skills matched.")
    elif coverage >= 50:
        explanations.append(f"Partial skill match ({coverage}% of required skills met).")
    elif coverage > 0:
        explanations.append(f"Low skill match ({coverage}% of required skills met).")
    else:
        explanations.append("No required skills matched.")
        
    # Experience gap
    exp_gap = features_dict.get('experience_gap', 0)
    if exp_gap > 0:
        explanations.append(f"Experience requirement satisfied (exceeds by {exp_gap} years).")
    elif exp_gap == 0:
        explanations.append("Experience requirement exactly satisfied.")
    else:
        explanations.append(f"Missing {-exp_gap} years of required experience.")
        
    # Education match
    edu_match = features_dict.get('education_match', 0)
    if edu_match == 1:
        explanations.append("Preferred education level met or exceeded.")
    else:
        explanations.append("Did not meet preferred education level.")
        
    return explanations
