import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_logistic_regression(data_path="data/features.csv", model_path="models/model.pkl"):
    """
    Trains a Logistic Regression model on the engineered features.
    """
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    if 'successful_match' not in df.columns:
        print("Target variable 'successful_match' not found.")
        return
        
    features = ['skill_overlap', 'experience_match', 'education_match', 'location_match']
    X = df[features]
    y = df['successful_match']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model trained and saved to {model_path}")
    
    # Save test set for evaluation later to keep it isolated
    test_df = X_test.copy()
    test_df['successful_match'] = y_test
    # add back the ids for evaluation if needed
    test_df['student_id'] = df.loc[test_df.index, 'student_id']
    test_df['job_id'] = df.loc[test_df.index, 'job_id']
    test_df.to_csv("data/test_set.csv", index=False)
    print("Test set saved to data/test_set.csv for evaluation.")

if __name__ == "__main__":
    train_logistic_regression()
