import os
import json
import uuid
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

from features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODELS_DIR = os.path.join(BASE_DIR, '../models')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, '../experiments')

def load_data(split='train'):
    pairs = pd.read_csv(os.path.join(DATA_DIR, f'{split}_pairs.csv'))
    candidates = pd.read_csv(os.path.join(DATA_DIR, 'candidates.csv'))
    jobs = pd.read_csv(os.path.join(DATA_DIR, 'jobs.csv'))
    
    # Merge
    data = pairs.merge(candidates, on='Candidate ID').merge(jobs, on='Job ID')
    return data

def build_features(data_df):
    X = []
    y = data_df['is_match'].values
    
    for _, row in data_df.iterrows():
        cand = {
            'Skills': row['Skills'],
            'Experience Years': row['Experience Years'],
            'Education': row['Education'],
            'Certifications': row['Certifications'],
            'Projects': row['Projects']
        }
        job = {
            'Required Skills': row['Required Skills'],
            'Preferred Skills': row['Preferred Skills'],
            'Experience Requirement': row['Experience Requirement'],
            'Education Requirement': row['Education Requirement']
        }
        feats = extract_features(cand, job)
        X.append([
            feats['skill_match_pct'],
            feats['req_skill_coverage'],
            feats['pref_skill_coverage'],
            feats['exp_match'],
            feats['edu_match'],
            feats['cert_match'],
            feats['project_relevance']
        ])
        
    return np.array(X), y

def main():
    print("Loading data...")
    train_df = load_data('train')
    val_df = load_data('val')
    
    print("Building features...")
    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr
    }
    
    print(f"Metrics: {metrics}")
    
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'logistic_regression.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Log experiment
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    run_id = str(uuid.uuid4())
    log_entry = {
        'run_id': run_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'model': 'LogisticRegression',
        'hyperparameters': {
            'class_weight': 'balanced'
        },
        'features': [
            'skill_match_pct', 'req_skill_coverage', 'pref_skill_coverage',
            'exp_match', 'edu_match', 'cert_match', 'project_relevance'
        ],
        'metrics': metrics
    }
    
    log_file = os.path.join(EXPERIMENTS_DIR, 'runs.jsonl')
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f"Experiment logged to {log_file}")

if __name__ == '__main__':
    main()
