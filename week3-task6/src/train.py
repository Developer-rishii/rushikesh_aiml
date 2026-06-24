import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from feature_engineering import extract_features

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    models_dir = os.path.join(current_dir, "..", "models")
    
    candidates_df, jobs_df = load_data(data_dir)
    
    print("Extracting features (this may take a moment)...")
    features_df = extract_features(candidates_df, jobs_df)
    
    # Save the feature table so evaluate.py doesn't re-compute it, or just rely on reproducible split.
    # Better to save the test split IDs to ensure evaluate.py uses exactly the held-out test split.
    
    X = features_df[['skill_overlap_percentage', 'experience_gap', 'education_match', 'certification_match_count', 'required_skill_coverage']]
    y = features_df['label']
    
    # 70% train / 15% validation / 15% test
    # First split off 15% for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Then split the remaining 85% into 70/15 -> approx 17.65% of 85 is 15
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Logistic Regression...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "baseline_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
