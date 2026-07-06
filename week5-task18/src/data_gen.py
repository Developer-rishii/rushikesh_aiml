import pandas as pd
import numpy as np
import json
import os
import random

def generate_rec_v1_output():
    np.random.seed(42)
    random.seed(42)
    n_rows = 500
    
    colleges = [f"C{i:03d}" for i in range(1, 6)]
    students = [f"S{i:04d}" for i in range(1, 101)]
    jobs = [f"J{i:03d}" for i in range(1, 21)]
    skills = ["Python", "SQL", "Docker", "AWS", "Java", "React", "K8s", "Git"]
    
    data = []
    for _ in range(n_rows):
        college_id = np.random.choice(colleges)
        student_id = np.random.choice(students)
        job_id = np.random.choice(jobs)
        rank_position = np.random.randint(1, 6)
        match_score = np.round(np.random.uniform(0.5, 0.95), 2)
        predicted_relevance_score = np.round(match_score * 0.9 + np.random.uniform(-0.05, 0.05), 3)
        skill_overlap_count = np.random.randint(2, 6)
        skill_gap_count = np.random.randint(0, 4)
        
        if rank_position == 1 and np.random.rand() > 0.3:
            skill_gap_count = 0
            
        gap_skills = np.random.choice(skills, size=skill_gap_count, replace=False).tolist() if skill_gap_count > 0 else []
        skill_gap_list = ",".join(gap_skills)
        years_exposure_avg = np.round(np.random.uniform(0, 5), 1)
        jd_seniority_num = np.random.choice([1, 2, 3, 4, 5])
        
        ai_trust_score = np.round(np.random.uniform(0.7, 0.99), 2)
        verified_skill_count = np.random.randint(2, 10)
        
        feature_importances = {
            "match_score": 0.41,
            "skill_overlap_count": 0.28,
            "ai_trust_score": 0.19,
            "skill_gap_ratio": 0.12
        }
        feature_importances_json = json.dumps(feature_importances)
        
        if np.random.rand() > 0.5:
            task16_explanation = "Good match overall for this role based on profile."
        else:
            task16_explanation = f"Student matches well. Rank {rank_position}."
            
        if np.random.rand() < 0.02:
            feature_importances_json = None
            
        data.append([
            student_id, college_id, job_id, rank_position, match_score, 
            predicted_relevance_score, skill_overlap_count, skill_gap_count, 
            skill_gap_list, years_exposure_avg, jd_seniority_num, 
            ai_trust_score, verified_skill_count, feature_importances_json, task16_explanation
        ])
        
    cols = ["student_id", "college_id", "job_id", "rank_position", "match_score", 
            "predicted_relevance_score", "skill_overlap_count", "skill_gap_count", 
            "skill_gap_list", "years_exposure_avg", "jd_seniority_level", 
            "ai_trust_score", "verified_skill_count", "feature_importances_json", "task16_explanation"]
            
    df = pd.DataFrame(data, columns=cols)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/rec_v1_output.csv", index=False)
    print("Created data/rec_v1_output.csv")

def generate_explanation_quality_labels():
    np.random.seed(42)
    random.seed(42)
    data = []
    students = [f"S{i:04d}" for i in range(1, 201)]
    jobs = [f"J{i:03d}" for i in range(1, 21)]
    skills_list = ["Python", "SQL", "Docker", "AWS", "Java", "React", "K8s", "Git", "C++", "Ruby", "Go", "Azure"]
    
    # Generate 250 distinct rows
    for i in range(250):
        student = np.random.choice(students)
        job = np.random.choice(jobs)
        
        true_rank = np.random.randint(2, 6)
        true_gaps = np.random.choice(skills_list, size=np.random.randint(1, 3), replace=False).tolist()
        skill_gap_list = ",".join(true_gaps)
        
        is_good = np.random.rand() > 0.4
        
        if is_good:
            struct_type = np.random.randint(0, 4)
            if struct_type == 0:
                text = f"You're ranked #{true_rank} for this role. Missing: {true_gaps[0]}. Acquiring this skill could move you to #{true_rank - 1}."
            elif struct_type == 1:
                text = f"Top features: match_score (0.41). Missing {true_gaps[0]}. Rank change if {true_gaps[0]} added: #{true_rank} -> #{true_rank - 1}."
            elif struct_type == 2:
                text = f"Your current position is #{true_rank}. Gaps identified: {true_gaps[0]}. If you learn {true_gaps[0]}, you might jump to #{true_rank - 1}."
            else:
                text = f"Rank: {true_rank}. {true_gaps[0]} is required. Adding it could shift you up to {true_rank - 1}."
            label = 1
            reason = "actionable, specific, mentions correct skills and rank"
        else:
            err_type = np.random.randint(0, 5)
            if err_type == 0:
                text = "Good match overall. Student is recommended."
                reason = "too generic"
            elif err_type == 1:
                wrong_skill = np.random.choice([s for s in skills_list if s not in true_gaps])
                text = f"You are rank #{true_rank}. Missing: {wrong_skill}. Add it to improve."
                reason = "hallucinated skill not in gap list"
            elif err_type == 2:
                wrong_rank = true_rank + 2
                text = f"You're ranked #{wrong_rank} for this role. Missing: {true_gaps[0]}."
                reason = "states incorrect rank"
            elif err_type == 3:
                text = f"Student has match_score weight 0.41 and ai_trust_score 0.99. Rank #{true_rank}."
                reason = "correct but not actionable for student, no counterfactual"
            else:
                text = f"You are rank #{true_rank}. You are missing {true_gaps[0]}. There is no way to improve."
                reason = "correct features but unhelpful and negative tone"
            label = 0
            
        audience = np.random.choice(['student', 'officer', 'admin'])
        
        data.append([student, job, text, label, reason, skill_gap_list, true_rank, audience])
        
    df = pd.DataFrame(data, columns=["student_id", "job_id", "explanation_text", "quality_label", "quality_reason", "skill_gap_list", "rank_position", "audience"])
    df.to_csv("data/explanation_quality_labels.csv", index=False)
    print("Created data/explanation_quality_labels.csv")

if __name__ == "__main__":
    generate_rec_v1_output()
    generate_explanation_quality_labels()
