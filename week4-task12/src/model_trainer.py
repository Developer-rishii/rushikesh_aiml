"""
Stage 3b: Train a binary classifier to filter candidates from Stage 3a.
A candidate is labeled as true positive if its canonical_name is in the 
document's ground-truth skill list.

Features:
  - match_type (one-hot: exact, alias, fuzzy)
  - fuzzy_score (float 0-100)
  - is_negated (bool -> int)
  - is_short_token (bool -> int)
  - token_length (int)
"""
import json
import os
import sys
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Add parent dir so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.ontology import SkillsOntology
from src.rule_extractor import extract_candidates


def featurize(candidate):
    """Convert a candidate dict into a feature vector."""
    return {
        'match_exact': 1 if candidate['match_type'] == 'exact' else 0,
        'match_alias': 1 if candidate['match_type'] == 'alias' else 0,
        'match_fuzzy': 1 if candidate['match_type'] == 'fuzzy' else 0,
        'fuzzy_score': candidate['fuzzy_score'] / 100.0,
        'is_negated': int(candidate['is_negated']),
        'is_short_token': int(candidate['is_short_token']),
        'token_length': candidate['token_length'],
    }


FEATURE_NAMES = ['match_exact', 'match_alias', 'match_fuzzy', 'fuzzy_score',
                 'is_negated', 'is_short_token', 'token_length']


def build_training_data(ontology):
    """
    Run Stage 3a on all sample documents, label each candidate as 
    true positive (1) or false positive (0) using ground-truth.
    """
    rows = []
    
    for data_file, doc_type in [('data/resumes.json', 'resume'), ('data/jds.json', 'jd')]:
        with open(data_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        for doc in docs:
            doc_id = doc['doc_id']
            text = doc['text']
            gt_skills = set(doc['ground_truth'])
            
            candidates = extract_candidates(text, ontology)
            
            for cand in candidates:
                features = featurize(cand)
                label = 1 if cand['canonical_name'] in gt_skills else 0
                features['label'] = label
                features['doc_id'] = doc_id
                features['doc_type'] = doc_type
                features['canonical_name'] = cand['canonical_name']
                rows.append(features)
    
    return pd.DataFrame(rows)


def train_model(ontology_path='data/skills_ontology.csv', model_output_path='src/models/skill_classifier.pkl'):
    """Train the ML confidence model and save to disk."""
    ontology = SkillsOntology(ontology_path)
    
    print("Building training data from sample documents...")
    df = build_training_data(ontology)
    
    if df.empty:
        print("ERROR: No training data generated. Check ontology and sample data.")
        return None
    
    print(f"Total candidates: {len(df)}, True Positives: {df['label'].sum()}, "
          f"False Positives: {(df['label'] == 0).sum()}")
    
    # Split by document (GroupShuffleSplit) to avoid leaking same-document candidates
    X = df[FEATURE_NAMES].values
    y = df['label'].values
    groups = df['doc_id'].values
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Further split train into train/val
    train_groups = groups[train_idx]
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sub_train_idx, val_idx = next(val_splitter.split(X_train, y_train, train_groups))
    
    X_sub_train, X_val = X_train[sub_train_idx], X_train[val_idx]
    y_sub_train, y_val = y_train[sub_train_idx], y_train[val_idx]
    
    print(f"Train: {len(X_sub_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_sub_train, y_sub_train)
    
    # Evaluate
    train_acc = accuracy_score(y_sub_train, model.predict(X_sub_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    val_prec = precision_score(y_val, model.predict(X_val), zero_division=0)
    val_rec = recall_score(y_val, model.predict(X_val), zero_division=0)
    test_prec = precision_score(y_test, model.predict(X_test), zero_division=0)
    test_rec = recall_score(y_test, model.predict(X_test), zero_division=0)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    
    # Feature importances
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))
    print(f"\nFeature importances: {importances}")
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")
    
    # Log experiment
    os.makedirs('experiments', exist_ok=True)
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'params': {'n_estimators': 100, 'max_depth': 5, 'class_weight': 'balanced'},
        'train_size': len(X_sub_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_accuracy': round(train_acc, 4),
        'val_accuracy': round(val_acc, 4),
        'val_precision': round(val_prec, 4),
        'val_recall': round(val_rec, 4),
        'test_accuracy': round(test_acc, 4),
        'test_precision': round(test_prec, 4),
        'test_recall': round(test_rec, 4),
        'feature_importances': {k: round(v, 4) for k, v in importances.items()},
        'model_path': model_output_path,
    }
    
    log_path = 'experiments/training_log.jsonl'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f"Experiment logged to {log_path}")
    
    return model, df


if __name__ == '__main__':
    train_model()
