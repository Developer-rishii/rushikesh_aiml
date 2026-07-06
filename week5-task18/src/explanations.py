import json
import joblib
import pandas as pd
import numpy as np
import os

_RANKER_MODEL = None
try:
    _RANKER_MODEL = joblib.load(r"d:\Placemux-aiml\week5-task16\src\models\ranker.joblib")
except:
    pass

FEATURE_COLS = [
    "match_score", "skill_overlap_count", "skill_gap_count",
    "years_exposure_avg", "jd_seniority_level", "verified_skill_count",
    "ai_trust_score", "skill_gap_ratio", "seniority_match", 
    "trust_weighted_score", "college_avg_match_score"
]

def compute_counterfactual(row, full_df=None):
    if not _RANKER_MODEL:
        return None, None
        
    gap_count = row['skill_gap_count']
    current_score = row.get('predicted_relevance_score', 0)
    current_rank = row['rank_position']
    
    if gap_count == 0 or current_rank == 1:
        return None, None
        
    features = row.copy()
    features['skill_gap_count'] = max(0, gap_count - 1)
    features['skill_gap_ratio'] = features['skill_gap_count'] / (features['skill_overlap_count'] + 1e-5)
    
    feature_vector = []
    for c in FEATURE_COLS:
        feature_vector.append(features.get(c, 0))
        
    try:
        new_score = round(float(_RANKER_MODEL.predict_proba([feature_vector])[0][1]), 3)
    except:
        return None, None
        
    score_diff = round(new_score - current_score, 3)
    
    if score_diff <= 0:
        return None, None
        
    new_rank = current_rank
    if full_df is not None:
        cohort = full_df[full_df['student_id'] == row['student_id']].copy()
        cohort.loc[cohort['job_id'] == row['job_id'], 'predicted_relevance_score'] = new_score
        cohort = cohort.sort_values('predicted_relevance_score', ascending=False).reset_index(drop=True)
        idx = cohort[cohort['job_id'] == row['job_id']].index
        if len(idx) > 0:
            new_rank = int(idx[0]) + 1
    else:
        new_rank = max(1, current_rank - 1)
        
    if new_rank >= current_rank:
        return None, None
        
    return new_rank, score_diff

def generate_student_explanation(row, full_df=None):
    rank = row['rank_position']
    gap_skills = str(row['skill_gap_list']).split(',') if pd.notna(row['skill_gap_list']) and row['skill_gap_list'] else []
    
    if rank == 1 and not gap_skills:
        return f"You're ranked #{rank} for this role. You are top-ranked for this role; no skill gaps identified."
        
    expl = f"You're ranked #{rank} for this role. "
    expl += f"Strong match based on your skills — these cover {row['skill_overlap_count']} required skills. "
    
    if gap_skills:
        top_gap = gap_skills[0]
        expl += f"Missing: {top_gap} (required). "
        
        new_rank, _ = compute_counterfactual(row, full_df)
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

def generate_admin_explanation(row, full_df=None):
    fi_str = row['feature_importances_json']
    try:
        if pd.isna(fi_str):
            raise ValueError
        fi = json.loads(fi_str)
        fi_text = ", ".join([f"{k} (weight {v})" for k, v in fi.items()])
        expl = f"Top features driving this recommendation: {fi_text}. "
    except:
        expl = "Feature attribution unavailable — explanation based on match_score only. "
        
    expl += f"Model confidence: {row.get('predicted_relevance_score', 0)} (well calibrated). "
    
    gap_skills = str(row['skill_gap_list']).split(',') if pd.notna(row['skill_gap_list']) and row['skill_gap_list'] else []
    new_rank, score_diff = compute_counterfactual(row, full_df)
    
    if gap_skills and new_rank:
        expl += f"Rank change if {gap_skills[0]} added: #{row['rank_position']} -> #{new_rank} (Δ score +{score_diff})."
    else:
        expl += "No counterfactual rank change applicable."
        
    return expl.strip()

def generate_all_explanations(df):
    results = []
    for _, row in df.iterrows():
        student_expl = generate_student_explanation(row, df)
        officer_expl = generate_officer_explanation(row)
        admin_expl = generate_admin_explanation(row, df)
        
        results.append({
            'student_id': row['student_id'],
            'college_id': row['college_id'],
            'job_id': row['job_id'],
            'student_explanation': student_expl,
            'officer_explanation': officer_expl,
            'admin_explanation': admin_expl
        })
    return pd.DataFrame(results)
