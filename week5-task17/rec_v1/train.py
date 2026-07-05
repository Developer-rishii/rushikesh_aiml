"""
train.py

Trains the ML model, sweeps hyperparameters, logs experiments, and saves the best model.
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score
from features import FeatureEngineer

def train_pipeline():
    print("=== Training Pipeline ===")
    
    fe = FeatureEngineer()
    fe.load_context()
    
    outcomes = pd.read_csv("data/outcomes.csv")
    
    # 1. Split (Train 70%, Val 15%, Test 15%)
    # Grouping by student_id prevents a student's applications leaking across folds
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(outcomes, groups=outcomes["student_id"]))
    
    train_outcomes = outcomes.iloc[train_idx]
    temp_outcomes = outcomes.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_outcomes, groups=temp_outcomes["student_id"]))
    
    val_outcomes = temp_outcomes.iloc[val_idx]
    test_outcomes = temp_outcomes.iloc[test_idx]
    
    print(f"Data Split: Train={len(train_outcomes)}, Val={len(val_outcomes)}, Test={len(test_outcomes)}")
    
    # Save test set for evaluate.py
    test_outcomes.to_csv("data/test_outcomes.csv", index=False)
    
    # 2. Extract Features
    print("Engineering features...")
    fe.fit(train_outcomes) # Calculate priors on train only
    
    X_train = fe.transform(train_outcomes)
    y_train = train_outcomes["was_hired"].values
    
    X_val = fe.transform(val_outcomes)
    y_val = val_outcomes["was_hired"].values
    
    features_to_use = ["skill_overlap_ratio", "proficiency_gap", "experience_fit", "college_hire_prior"]
    
    X_train_sub = X_train[features_to_use]
    X_val_sub = X_val[features_to_use]
    
    # 3. Hyperparameter Sweep
    configs = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.2},
    ]
    
    best_auc = 0
    best_model = None
    best_config = None
    
    print("Running Hyperparameter Sweep...")
    with open("experiments.jsonl", "w") as f:
        for i, config in enumerate(configs):
            model = GradientBoostingClassifier(**config, random_state=42)
            model.fit(X_train_sub, y_train)
            
            y_val_pred_proba = model.predict_proba(X_val_sub)[:, 1]
            y_val_pred = (y_val_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_val_pred_proba)
            prec = precision_score(y_val, y_val_pred, zero_division=0)
            
            log_entry = {
                "run": i,
                "params": config,
                "val_auc": auc,
                "val_precision": prec
            }
            f.write(json.dumps(log_entry) + "\\n")
            print(f"Run {i}: AUC={auc:.4f}, Prec={prec:.4f} | {config}")
            
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_config = config
                
    print(f"\\nBest Model AUC: {best_auc:.4f} with params: {best_config}")
    
    # 4. Persist
    artifact = {
        "model": best_model,
        "features_to_use": features_to_use,
        "college_priors": fe.priors # Must persist the priors computed during train
    }
    joblib.dump(artifact, "model.pkl")
    print("Saved best model to model.pkl")

if __name__ == "__main__":
    train_pipeline()
