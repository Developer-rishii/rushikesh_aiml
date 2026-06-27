import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import joblib
import os

def engineer_features(df):
    """
    Creates feature matrix X from the raw dataframe.
    Features: 
    - overlap_ratio: (overlap / required)
    - missing_skills_count: (required - overlap)
    - seniority_gap: (student_years_exp - job_seniority)
    - years_exp: student_years_exp
    - job_seniority: job_seniority
    """
    X = pd.DataFrame(index=df.index)
    X['overlap_ratio'] = df['overlap_ratio']
    
    missing = []
    for i, row in df.iterrows():
        overlap = len(set(row['student_skills']).intersection(set(row['job_skills'])))
        total_req = len(row['job_skills'])
        missing.append(total_req - overlap)
        
    X['missing_skills_count'] = missing
    X['seniority_gap'] = df['student_years_exp'] - df['job_seniority']
    X['years_exp'] = df['student_years_exp']
    X['job_seniority'] = df['job_seniority']
    
    return X

class MatchModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()
        self.is_trained = True
        
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }, path)

    @classmethod
    def load(cls, path):
        instance = cls()
        data = joblib.load(path)
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.is_trained = data['is_trained']
        return instance

    def explain(self, row, features):
        """
        Provides a plain-English explanation of why this prediction was made.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
            
        prob = self.model.predict_proba(features)[0, 1]
        decision = "Good Match" if prob >= 0.5 else "Poor Match"
        
        # Get coefficients
        coefs = self.model.coef_[0]
        
        # Calculate feature contributions
        contributions = coefs * features.iloc[0].values
        
        # Top positive and negative contributors
        feat_contrib = list(zip(self.feature_names, contributions))
        feat_contrib.sort(key=lambda x: x[1], reverse=True)
        
        pos_reasons = []
        neg_reasons = []
        
        for feat, val in feat_contrib:
            if val > 0.2:
                pos_reasons.append(feat)
            elif val < -0.2:
                neg_reasons.append(feat)
                
        explanation = f"Score: {prob:.2f} ({decision}). "
        
        # Include specific skills context
        req_skills = set(row['job_skills'].iloc[0])
        student_skills = set(row['student_skills'].iloc[0])
        matched = req_skills.intersection(student_skills)
        missing = req_skills - student_skills
        
        explanation += f"Matched on {len(matched)}/{len(req_skills)} required skills: {', '.join(matched) if matched else 'None'}. "
        if missing:
            explanation += f"Missing: {', '.join(missing)}. "
            
        if pos_reasons:
            explanation += f"Model weighted {', '.join(pos_reasons)} positively. "
        if neg_reasons:
            explanation += f"Model penalized by {', '.join(neg_reasons)}."
            
        return explanation

def evaluate_model(model, X, y, split_name="Test"):
    y_pred = model.predict(X)
    
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'fpr': fpr
    }
    
    print(f"[{split_name} Model] Precision: {precision:.3f}, Recall: {recall:.3f}, FPR: {fpr:.3f}")
    return metrics
