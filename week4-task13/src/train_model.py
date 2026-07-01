import os
import pandas as pd
import numpy as np
import pickle
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
from xgboost import XGBClassifier
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.generate_synthetic_sessions import load_data

def preprocess_and_split(df):
    """
    Splits the data 70/15/15 (train/val/test) stratified by label.
    """
    # Clean up any complete garbage rows
    # Remember our adversarial malformed row where tab_switch_count = -999?
    # We shouldn't remove it here if we want edge-case handling in inference,
    # but for training we want clean data. We'll clip it or keep it.
    
    X = df.drop(columns=['session_id', 'label'])
    y = (df['label'] == 'true_violation').astype(int)
    
    # 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # 15% Val, 15% Test (Split Temp 50/50)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    # Define preprocessing steps
    num_features = ['tab_switch_count', 'face_absent_seconds', 'multiple_faces_detected', 
                    'audio_anomaly_score', 'eye_gaze_offscreen_pct', 'session_duration', 
                    'candidate_history_flag_rate']
    cat_features = ['device_type', 'network_quality', 'time_of_day']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def log_experiment(model_name, params, val_loss, val_f1):
    log_file = os.path.join(os.path.dirname(__file__), '..', 'experiment_log.csv')
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model', 'params', 'val_log_loss', 'val_f1'])
        writer.writerow([datetime.now().isoformat(), model_name, str(params), val_loss, val_f1])

def train():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_sessions.csv')
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print("Data not found. Please generate data first.")
        return
        
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_and_split(df)
    
    print("Training Logistic Regression...")
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_val)
    lr_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_loss = log_loss(y_val, lr_probs)
    lr_f1 = f1_score(y_val, lr_preds)
    log_experiment('Logistic Regression', {'class_weight': 'balanced'}, lr_loss, lr_f1)
    
    print("Training XGBoost...")
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=2.33))])
    # 70% FP, 30% TP -> weight ~ 70/30 = 2.33
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_val)
    xgb_probs = xgb_pipeline.predict_proba(X_val)[:, 1]
    xgb_loss = log_loss(y_val, xgb_probs)
    xgb_f1 = f1_score(y_val, xgb_preds)
    log_experiment('XGBoost', {'scale_pos_weight': 2.33}, xgb_loss, xgb_f1)
    
    print(f"LR Val F1: {lr_f1:.3f} | XGB Val F1: {xgb_f1:.3f}")
    
    # Save best model
    best_pipeline = xgb_pipeline if xgb_f1 > lr_f1 else lr_pipeline
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved best model to {model_path}")

    # Save test set for later evaluation (to ensure no leakage)
    test_set = pd.concat([X_test, y_test.rename('true_violation')], axis=1)
    test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_set.csv')
    test_set.to_csv(test_path, index=False)
    print(f"Saved test set to {test_path}")

if __name__ == "__main__":
    train()
