import pandas as pd
import numpy as np

def calculate_baseline_score(df_features):
    """
    Calculates the baseline score using the formula:
    0.7 * skill_overlap + 0.2 * experience_match + 0.1 * education_match
    """
    df = df_features.copy()
    df['baseline_score'] = (
        0.7 * df['skill_overlap'] +
        0.2 * df['experience_match'] +
        0.1 * df['education_match']
    )
    return df

def evaluate_baseline(df):
    """
    Evaluates the baseline by treating score >= 0.5 as a positive prediction.
    """
    if 'successful_match' not in df.columns:
        print("Cannot evaluate baseline without target labels.")
        return
        
    df['baseline_pred'] = (df['baseline_score'] >= 0.5).astype(int)
    
    # Calculate simple metrics
    correct = (df['baseline_pred'] == df['successful_match']).sum()
    accuracy = correct / len(df)
    print(f"Baseline Accuracy (threshold=0.5): {accuracy:.4f}")
    
    return df

if __name__ == "__main__":
    df_features = pd.read_csv("data/features.csv")
    df_scored = calculate_baseline_score(df_features)
    evaluate_baseline(df_scored)
    
    # Save baseline predictions for evaluation comparison later
    df_scored.to_csv("data/baseline_results.csv", index=False)
    print("Baseline scored and saved to data/baseline_results.csv")
