import pandas as pd

def calculate_completeness(explanation: str, has_skill: bool, rank_position: int) -> float:
    score = 0.0
    text_lower = explanation.lower()
    
    # Names top feature? For simplicity, we check if it mentions 'weight' or 'feature' or 'match'
    if "weight" in text_lower or "feature" in text_lower or "match" in text_lower:
        score += 0.33
        
    # Names a specific skill?
    if has_skill or any(s in text_lower for s in ["python", "sql", "docker", "aws", "java", "react", "k8s", "git"]):
        score += 0.33
        
    # Names rank position?
    if "rank" in text_lower or f"#{rank_position}" in text_lower:
        score += 0.34
        
    return score

def calculate_actionability(explanation: str) -> float:
    text_lower = explanation.lower()
    
    # Proxy: specific skill gap AND a counterfactual (e.g. rank improves, move you to)
    has_gap = "missing" in text_lower or "gap" in text_lower
    has_counterfactual = "improve" in text_lower or "move you" in text_lower or "change" in text_lower or "added" in text_lower
    
    if has_gap and has_counterfactual:
        return 1.0
    return 0.0

def calculate_specificity(explanation: str) -> float:
    # Does it use specific numbers (score, rank, skill count)?
    # We check for digits
    import re
    digits = re.findall(r'\d+', explanation)
    if len(digits) >= 1 and not ("good match overall" in explanation.lower() and len(digits) == 0):
        return 1.0
    return 0.0

def evaluate_baseline(df):
    results = []
    for _, row in df.iterrows():
        expl = str(row['task16_explanation'])
        rank = row['rank_position']
        skill_list = str(row['skill_gap_list']).lower() if pd.notna(row['skill_gap_list']) else ""
        has_skill = any(s in expl.lower() for s in skill_list.split(",")) if skill_list else False
        
        comp = calculate_completeness(expl, has_skill, rank)
        act = calculate_actionability(expl)
        spec = calculate_specificity(expl)
        
        results.append({
            'completeness_score': comp,
            'actionability_score': act,
            'specificity_score': spec
        })
    return pd.DataFrame(results)
