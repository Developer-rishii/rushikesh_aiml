import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import re

def extract_features(row):
    text = str(row.get('explanation_text', row.get('task16_explanation', ''))).lower()
    
    explanation_length_tokens = len(text.split())
    
    gap_skills = str(row.get('skill_gap_list', '')).lower().split(',')
    gap_skills = [s.strip() for s in gap_skills if s.strip()]
    
    mentioned_count = 0
    for s in gap_skills:
        if s and s in text:
            mentioned_count += 1
            
    true_rank = str(row.get('rank_position', '3'))
    rank_matches = 1 if true_rank in text else 0
    
    has_numeric = int(bool(re.search(r'\d+', text)))
    
    audience = row.get('audience', 'student')
    audience_score = 1
    if audience == 'student' and ("weight" in text or "feature" in text):
        audience_score = 0
        
    return {
        'explanation_length_tokens': explanation_length_tokens,
        'num_distinct_skills_mentioned': mentioned_count,
        'rank_matches_true_rank': rank_matches,
        'has_numeric': has_numeric,
        'audience_alignment': audience_score
    }

def train_model():
    df = pd.read_csv("data/explanation_quality_labels.csv")
    
    X_features = pd.DataFrame([extract_features(r) for _, r in df.iterrows()])
    y = df['quality_label']
    
    X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"Model trained. Validation accuracy: {acc:.2f}")
    
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(model, "src/models/explanation_quality_scorer.joblib")
    print("Saved model to src/models/explanation_quality_scorer.joblib")

def score_explanations(df, text_col='explanation_text'):
    model = joblib.load("src/models/explanation_quality_scorer.joblib")
    features = pd.DataFrame([extract_features(r) for _, r in df.iterrows()])
    probs = model.predict_proba(features)[:, 1]
    return probs

if __name__ == "__main__":
    train_model()
