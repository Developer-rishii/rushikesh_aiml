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
    
    has_specific_skill_named = any(s in text for s in ["python", "sql", "docker", "aws", "java", "react", "k8s", "git"])
    has_counterfactual = any(w in text for w in ["improve", "change", "move you to", "added"])
    has_numeric_score = bool(re.search(r'\d+\.\d+', text))
    
    # Simple completeness and specificity
    completeness_score = 0.0
    if "weight" in text or "feature" in text or "match" in text: completeness_score += 0.33
    if has_specific_skill_named: completeness_score += 0.33
    if "rank" in text or "#" in text: completeness_score += 0.34
    
    specificity_score = 1.0 if bool(re.search(r'\d+', text)) else 0.0
    
    audience = row.get('audience', 'student')
    audience_alignment_score = 1
    if audience == 'student' and ("weight" in text or "feature_importances" in text):
        audience_alignment_score = 0
    if audience == 'admin' and "weight" not in text:
        audience_alignment_score = 0
        
    explanation_length_tokens = len(text.split())
    rank_position = row.get('rank_position', 3)
    
    return {
        'has_specific_skill_named': int(has_specific_skill_named),
        'has_counterfactual': int(has_counterfactual),
        'has_numeric_score': int(has_numeric_score),
        'completeness_score': completeness_score,
        'specificity_score': specificity_score,
        'audience_alignment_score': audience_alignment_score,
        'explanation_length_tokens': explanation_length_tokens,
        'rank_position': rank_position
    }

def train_model():
    df = pd.read_csv("data/explanation_quality_labels.csv")
    df['rank_position'] = np.random.randint(1, 6, size=len(df))
    df['audience'] = np.random.choice(['student', 'officer', 'admin'], size=len(df))
    
    X_features = pd.DataFrame([extract_features(r) for _, r in df.iterrows()])
    y = df['quality_label']
    
    X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
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
