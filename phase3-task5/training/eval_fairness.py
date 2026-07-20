import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def eval_fairness():
    print("Evaluating fairness on test set...")
    df = pd.read_csv('data/interactions.csv')
    
    unique_candidates = df['candidate_id'].unique()
    _, test_cands = train_test_split(unique_candidates, test_size=0.2, random_state=42)
    test_df = df[df['candidate_id'].isin(test_cands)].copy()

    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
    X_test = test_df[features]
    
    # Load model
    model = lgb.Booster(model_file='models/lgbm_ranker.txt')
    
    test_df['predicted_score'] = model.predict(X_test)
    
    # Demographic Parity: Average predicted score for Group 0 vs Group 1
    # Note: the generative script baked in a negative bias for group 1.
    group0 = test_df[test_df['demographic_group'] == 0]
    group1 = test_df[test_df['demographic_group'] == 1]
    
    dp0 = group0['predicted_score'].mean()
    dp1 = group1['predicted_score'].mean()
    
    print(f"Demographic Parity Check:")
    print(f"  Group 0 Mean Score: {dp0:.4f}")
    print(f"  Group 1 Mean Score: {dp1:.4f}")
    print(f"  Difference: {dp0 - dp1:.4f} (Ideal is 0)")
    
    # Equal Opportunity: Average predicted score for ACTUALLY relevant items (relevance > 0)
    rel_group0 = group0[group0['relevance'] > 0]
    rel_group1 = group1[group1['relevance'] > 0]
    
    eo0 = rel_group0['predicted_score'].mean()
    eo1 = rel_group1['predicted_score'].mean()
    
    print(f"\nEqual Opportunity Check (Relevance > 0):")
    print(f"  Group 0 Mean Score: {eo0:.4f}")
    print(f"  Group 1 Mean Score: {eo1:.4f}")
    print(f"  Difference: {eo0 - eo1:.4f} (Ideal is 0)")

if __name__ == "__main__":
    eval_fairness()
