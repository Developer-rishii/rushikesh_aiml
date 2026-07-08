import pandas as pd

def generate_explanation(item_stats: dict, is_weak: bool, model_confidence: float = None) -> dict:
    response_count = item_stats.get('response_count', 0)
    
    if response_count < 20:
        return {
            "status": "needs_more_data",
            "admin_view": f"Item has only {response_count} responses. Need at least 20 to evaluate reliability.",
            "recruiter_view": "Not enough data to evaluate this question yet."
        }
        
    p_value = item_stats.get('p_value', 0)
    pb_corr = item_stats.get('point_biserial_corr', 0)
    bottom_rate = item_stats.get('bottom_25_correct_rate', 0)
    top_rate = item_stats.get('top_25_correct_rate', 0)
    subject = item_stats.get('subject', 'Unknown Subject')
    
    if not is_weak:
        return {
            "status": "strong",
            "admin_view": f"Item is performing well based on {response_count} responses. Discrimination: {pb_corr:.2f}, Difficulty: {p_value:.2f}.",
            "recruiter_view": f"This question in your {subject} test appears reliable."
        }
        
    # Weak Item Logic
    admin_reasons = []
    recruiter_reasons = []
    
    if p_value < 0.05:
        admin_reasons.append(f"extremely difficult (p-value {p_value:.3f} < 0.05)")
        recruiter_reasons.append("almost everyone gets it wrong, making it too hard to be useful")
    elif p_value > 0.95:
        admin_reasons.append(f"extremely easy (p-value {p_value:.3f} > 0.95)")
        recruiter_reasons.append("almost everyone gets it right, making it too easy to be useful")
        
    if pb_corr < 0.1:
        admin_reasons.append(f"poor/negative discrimination ({pb_corr:.2f}). Bottom 25% got it right {bottom_rate*100:.0f}% of the time; top 25% got it right {top_rate*100:.0f}% of the time")
        recruiter_reasons.append("students who do well on the rest of the test don't do better on this question. This usually means the answer key is wrong or the question is ambiguous")
        
    if not admin_reasons:
        admin_reasons.append(f"flagged by model pattern (confidence {model_confidence:.2f} if known)")
        recruiter_reasons.append("statistical patterns suggest this question may be unreliable")
        
    admin_view = f"Flagged: {'; '.join(admin_reasons)}. Based on {response_count} responses."
    recruiter_view = f"This question in your {subject} test may be unreliable — {'; '.join(recruiter_reasons)}. Recommend review before next cohort."
    
    return {
        "status": "weak",
        "admin_view": admin_view,
        "recruiter_view": recruiter_view
    }
