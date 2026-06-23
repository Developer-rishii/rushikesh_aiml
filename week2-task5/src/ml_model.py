import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ast
import os
import joblib
import json
import json
import argparse
from log_experiment import log_experiment

def simulate_ground_truth(skill_overlap_percentage, average_verified_skill_score, experience_match):
    score = (0.4 * (skill_overlap_percentage / 100.0)
             + 0.4 * (average_verified_skill_score / 100.0)
             + 0.2 * experience_match)
    score += np.random.normal(0, 0.08) # inject realistic noise
    return 1 if score >= 0.40 else 0

def load_data():
    jobs_df = pd.read_csv("data/jobs.csv")
    students_df = pd.read_csv("data/students.csv")
    return jobs_df, students_df

def create_applications_dataset(jobs_df, students_df):
    # Create a cartesian product or random subset of pairs
    # For simplicity, pair each job with 30 random students
    np.random.seed(42)
    pairs = []
    student_ids = students_df['student_id'].tolist()
    
    for _, job in jobs_df.iterrows():
        applicants = np.random.choice(student_ids, size=50, replace=False)
        for s_id in applicants:
            pairs.append({"job_id": job['job_id'], "student_id": s_id})
            
    apps_df = pd.DataFrame(pairs)
    apps_df = apps_df.merge(jobs_df, on="job_id").merge(students_df, on="student_id")
    return apps_df

def feature_engineering(apps_df, use_leaky_label=False):
    features = []
    targets = []
    
    for _, row in apps_df.iterrows():
        # Parse skills
        job_req = [s.strip().lower() for s in str(row['required_skills']).split(',') if s.strip()]
        student_skills = [s.strip().lower() for s in str(row['verified_skills']).split(',') if s.strip()]
        
        # Skill scores
        try:
            raw_skill_scores = ast.literal_eval(row['skill_scores'])
            skill_scores = {str(k).strip().lower(): v for k, v in raw_skill_scores.items()}
        except:
            skill_scores = {}
        
        # Calculate features
        matched_skills = [s for s in job_req if s in student_skills]
        matched_skill_count = len(matched_skills)
        missing_skill_count = len(job_req) - matched_skill_count
        skill_overlap_percentage = (matched_skill_count / len(job_req)) * 100 if len(job_req) > 0 else 0
        
        scores_for_matched = [skill_scores.get(s, 0) for s in matched_skills]
        # Average score of matched skills. If none matched, score is 0.
        average_verified_skill_score = np.mean(scores_for_matched) if scores_for_matched else 0.0
        
        experience_match = 1 if row['experience'] >= row['experience_required'] else 0
        
        features.append({
            "skill_overlap_percentage": skill_overlap_percentage,
            "matched_skill_count": matched_skill_count,
            "missing_skill_count": missing_skill_count,
            "average_verified_skill_score": average_verified_skill_score,
            "experience_match": experience_match
        })
        
        # Ground truth target:
        if use_leaky_label:
            is_match = 1 if (skill_overlap_percentage >= 70.0 and experience_match == 1) else 0
        else:
            is_match = simulate_ground_truth(skill_overlap_percentage, average_verified_skill_score, experience_match)
        targets.append(is_match)
        
    X = pd.DataFrame(features)
    y = np.array(targets)
    return X, y

def train_and_evaluate(X, y):
    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate on Test set
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    }
    
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaky", action="store_true", help="Use old leaky label")
    args = parser.parse_args()

    print("Loading data...")
    jobs_df, students_df = load_data()
    print("Creating dataset...")
    apps_df = create_applications_dataset(jobs_df, students_df)
    print(f"Total applications generated: {len(apps_df)}")
    
    print("Engineering features...")
    X, y = feature_engineering(apps_df, use_leaky_label=args.leaky)
    
    print("Training model...")
    model, metrics = train_and_evaluate(X, y)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if k in ["False Positive Rate", "False Negative Rate"]:
            print(f"{k} = {v*100:.2f}%")
        else:
            print(f"{k} = {v*100:.2f}%")
            
    # Print coefficients
    print("\nFeature Coefficients:")
    features_list = ["skill_overlap_percentage", "matched_skill_count", "missing_skill_count", "average_verified_skill_score", "experience_match"]
    for f_name, coef in zip(features_list, model.coef_[0]):
        print(f"  {f_name}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_regression.pkl")
    
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nModel saved to models/logistic_regression.pkl")

    # Log experiment
    notes = "BEFORE - leaky" if args.leaky else "AFTER - fixed"
    log_experiment(metrics, notes)
    print(f"Experiment logged: {notes}")
