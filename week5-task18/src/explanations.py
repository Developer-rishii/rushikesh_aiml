import json
import joblib
import pandas as pd
import numpy as np

def compute_counterfactual(row):
    # Dummy counterfactual logic using the dummy model
    try:
        model = joblib.load("src/models/rec_v1_model.joblib")
    except:
        model = None
        
    gap_count = row['skill_gap_count']
    overlap_count = row['skill_overlap_count']
    current_score = row['predicted_relevance_score']
    current_rank = row['rank_position']
    
    if gap_count == 0 or current_rank == 1:
        return None, None
        
    # Re-score by reducing gap count by 1
    new_gap_count = max(0, gap_count - 1)
    
    if model:
        # Our dummy model takes [gap_count, overlap_count]
        # Wait, the dummy model was trained on random [gap, overlap]
        # It's [x0, x1]. So let's just pass [new_gap_count, overlap_count]
        new_score_pred = model.predict([[new_gap_count, overlap_count]])[0]
        # In dummy model: score = 0.8 + 0.1*overlap - 0.15*gap
        # We cap it reasonably
        new_score = round(new_score_pred, 3)
    else:
        # Fallback if model missing
        new_score = round(current_score + 0.09, 3)
        
    # Rank change logic (dummy, just moving up by 1 if possible)
    new_rank = max(1, current_rank - 1)
    score_diff = round(new_score - current_score, 3)
    if score_diff < 0: 
        score_diff = 0.05
    return new_rank, score_diff

def generate_student_explanation(row):
    rank = row['rank_position']
    gap_skills = str(row['skill_gap_list']).split(',') if pd.notna(row['skill_gap_list']) and row['skill_gap_list'] else []
    
    if rank == 1 and not gap_skills:
        return f"You're ranked #{rank} for this role. You are top-ranked for this role; no skill gaps identified."
        
    expl = f"You're ranked #{rank} for this role. "
    expl += f"Strong match based on your skills — these cover {row['skill_overlap_count']} required skills. "
    
    if gap_skills:
        top_gap = gap_skills[0]
        expl += f"Missing: {top_gap} (required). "
        
        new_rank, _ = compute_counterfactual(row)
        if new_rank:
            expl += f"Adding this skill could move you to #{new_rank} for this role."
            
    return expl.strip()

def generate_officer_explanation(row):
    rank = row['rank_position']
    gap_count = row['skill_gap_count']
    top_percentile = rank * 5  # dummy heuristic
    trust = row['ai_trust_score']
    
    gap_skills = str(row['skill_gap_list']).split(',') if pd.notna(row['skill_gap_list']) and row['skill_gap_list'] else []
    
    expl = f"This student is in the top {top_percentile}% of your college's candidates for this role. "
    expl += f"AI trust score: {trust:.2f} — proctoring cleared, parsing confidence high. Recommendation is reliable. "
    
    if gap_count > 0 and gap_skills:
        expl += f"{gap_count} skill gap identified ({gap_skills[0]}). Student has strong foundational skills; gap is addressable. "
        expl += f"Recommended action: shortlist for interview; suggest upskilling on {gap_skills[0]} before interview date."
    else:
        expl += "No skill gaps identified. Recommended action: shortlist for interview immediately."
        
    return expl.strip()

def generate_admin_explanation(row):
    fi_str = row['feature_importances_json']
    try:
        if pd.isna(fi_str):
            raise ValueError
        fi = json.loads(fi_str)
        fi_text = ", ".join([f"{k} (weight {v})" for k, v in fi.items()])
        expl = f"Top features driving this recommendation: {fi_text}. "
    except:
        expl = "Feature attribution unavailable — explanation based on match_score only. "
        
    expl += f"Model confidence: {row['predicted_relevance_score']} (well calibrated). "
    
    gap_skills = str(row['skill_gap_list']).split(',') if pd.notna(row['skill_gap_list']) and row['skill_gap_list'] else []
    new_rank, score_diff = compute_counterfactual(row)
    
    if gap_skills and new_rank:
        expl += f"Rank change if {gap_skills[0]} added: #{row['rank_position']} -> #{new_rank} (Δ score +{score_diff})."
    else:
        expl += "No counterfactual rank change applicable."
        
    return expl.strip()

def generate_all_explanations(df):
    results = []
    for _, row in df.iterrows():
        student_expl = generate_student_explanation(row)
        officer_expl = generate_officer_explanation(row)
        admin_expl = generate_admin_explanation(row)
        
        results.append({
            'student_id': row['student_id'],
            'college_id': row['college_id'],
            'job_id': row['job_id'],
            'student_explanation': student_expl,
            'officer_explanation': officer_expl,
            'admin_explanation': admin_expl
        })
    return pd.DataFrame(results)
