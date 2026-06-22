import os
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.features.feature_engineering import build_feature_matrix
from src.evaluation.metrics import evaluate_and_log
from src.models.baseline import RuleBasedMatcher

def train_models():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df_students = pd.read_csv(os.path.join(data_dir, 'students.csv'))
    df_jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
    df_matches = pd.read_csv(os.path.join(data_dir, 'matches.csv'))
    
    # Fill NaN skills with empty string
    df_students['skills'] = df_students['skills'].fillna("")
    df_jobs['required_skills'] = df_jobs['required_skills'].fillna("")
    
    print("Building feature matrix...")
    X, y = build_feature_matrix(df_matches, df_students, df_jobs)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    run_id = int(time.time())
    
    # 1. Evaluate Baseline Model
    print("Evaluating Baseline Rule-Based Model...")
    baseline = RuleBasedMatcher()
    y_pred_baseline = []
    
    for _, row in X_test.iterrows():
        # Baseline relies on feature threshold for matches.
        # Let's say if overlap > 60, it's a match, just for evaluation purposes.
        # But wait, our baseline 'predict' returns the overlap score.
        score = row['skill_overlap_percentage']
        # we consider a match if overlap >= 50 for the baseline
        y_pred_baseline.append(1 if score >= 50 else 0)
        
    evaluate_and_log(run_id, "Baseline", y_test, y_pred_baseline)
    
    # 2. Train and Evaluate Logistic Regression
    print("Training Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    evaluate_and_log(run_id, "LogisticRegression", y_test, y_pred_lr)
    
    # Save Model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
        
    print("Models evaluated and Logistic Regression saved to models/model.pkl.")

if __name__ == "__main__":
    train_models()
