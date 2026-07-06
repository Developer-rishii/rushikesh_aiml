import pandas as pd
import numpy as np
import json
import joblib
from sklearn.linear_model import LinearRegression
import os

def generate_rec_v1_output():
    np.random.seed(42)
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
        
        # Rank 1 often has 0 skill gaps
        if rank_position == 1 and np.random.rand() > 0.3:
            skill_gap_count = 0
            
        gap_skills = np.random.choice(skills, size=skill_gap_count, replace=False).tolist() if skill_gap_count > 0 else []
        skill_gap_list = ",".join(gap_skills)
        years_exposure_avg = np.round(np.random.uniform(0, 5), 1)
        jd_seniority_level = np.random.choice(["Junior", "Mid", "Senior"])
        ai_trust_score = np.round(np.random.uniform(0.7, 0.99), 2)
        
        # We need realistic feature weights matching what we describe
        feature_importances = {
            "match_score": 0.41,
            "skill_overlap_count": 0.28,
            "ai_trust_score": 0.19,
            "skill_gap_ratio": 0.12
        }
        feature_importances_json = json.dumps(feature_importances)
        
        # Task 16 baseline explanation (can be generic or lack specifics)
        if np.random.rand() > 0.5:
            task16_explanation = "Good match overall for this role based on profile."
        else:
            task16_explanation = f"Student matches well. Rank {rank_position}."
            
        # Add edge case: missing feature_importances_json
        if np.random.rand() < 0.02:
            feature_importances_json = None
            
        data.append([
            student_id, college_id, job_id, rank_position, match_score, 
            predicted_relevance_score, skill_overlap_count, skill_gap_count, 
            skill_gap_list, years_exposure_avg, jd_seniority_level, 
            ai_trust_score, feature_importances_json, task16_explanation
        ])
        
    cols = ["student_id", "college_id", "job_id", "rank_position", "match_score", 
            "predicted_relevance_score", "skill_overlap_count", "skill_gap_count", 
            "skill_gap_list", "years_exposure_avg", "jd_seniority_level", 
            "ai_trust_score", "feature_importances_json", "task16_explanation"]
            
    df = pd.DataFrame(data, columns=cols)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/rec_v1_output.csv", index=False)
    print("Created data/rec_v1_output.csv")

def generate_explanation_quality_labels():
    np.random.seed(42)
    data = []
    students = [f"S{i:04d}" for i in range(1, 201)]
    jobs = [f"J{i:03d}" for i in range(1, 21)]
    
    examples = [
        # Good examples
        ("You're ranked #2 for this role. Missing: Docker. Adding this skill could move you to #1.", 1, "actionable and specific"),
        ("Top features: match_score (0.41). Missing Python. Rank change if Python added: #3 -> #2.", 1, "actionable and specific"),
        # Poor examples
        ("Good match overall", 0, "too generic"),
        ("Says skill X matters but model weighted Y highest", 0, "contradicts feature importances"),
        ("You are rank #4. Keep studying.", 0, "missing counterfactual"),
        ("Student has match_score weight 0.41 and ai_trust_score 0.99. Rank #2.", 0, "correct but not actionable for student")
    ]
    
    for i in range(200):
        student = np.random.choice(students)
        job = np.random.choice(jobs)
        ex = examples[i % len(examples)]
        data.append([student, job, ex[0], ex[1], ex[2]])
        
    df = pd.DataFrame(data, columns=["student_id", "job_id", "explanation_text", "quality_label", "quality_reason"])
    df.to_csv("data/explanation_quality_labels.csv", index=False)
    print("Created data/explanation_quality_labels.csv")

def generate_dummy_model():
    # We need a dummy model that takes something like [skill_gap_count, skill_overlap_count] and returns a score
    # So we can "re-score" counterfactuals
    X = np.random.rand(100, 2) * 5
    y = X[:, 1] * 0.1 - X[:, 0] * 0.15 + 0.8  # score = 0.8 + 0.1*overlap - 0.15*gap
    model = LinearRegression().fit(X, y)
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(model, "src/models/rec_v1_model.joblib")
    print("Created src/models/rec_v1_model.joblib")

if __name__ == "__main__":
    generate_rec_v1_output()
    generate_explanation_quality_labels()
    generate_dummy_model()
