import pandas as pd
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def explain_prediction():
    df = pd.read_csv('data/interactions.csv')
    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
    
    unique_candidates = df['candidate_id'].unique()
    _, test_cands = train_test_split(unique_candidates, test_size=0.2, random_state=42)
    test_df = df[df['candidate_id'].isin(test_cands)].copy()
    
    X_test = test_df[features]
    
    # Load model
    model = lgb.Booster(model_file='models/lgbm_ranker.txt')
    
    # Explain one prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Take the first candidate's first job prediction
    idx = 0
    candidate = test_df.iloc[idx]['candidate_id']
    job = test_df.iloc[idx]['job_id']
    
    base_value = explainer.expected_value
    if isinstance(base_value, list): # depending on SHAP/LGBM version
        base_value = base_value[0]
        
    prediction = model.predict(X_test.iloc[[idx]])[0]
    
    print(f"Explanation for Candidate: {candidate} matching with Job: {job}")
    print(f"Base Value (Average Prediction): {base_value:.4f}")
    print(f"Final Predicted Score: {prediction:.4f}")
    
    print("\nFeature Contributions:")
    contributions = []
    for i, feature in enumerate(features):
        val = X_test.iloc[idx][feature]
        contrib = shap_values[idx][i]
        contributions.append((feature, val, contrib))
        print(f"  - {feature} (value={val:.2f}): {contrib:+.4f}")
        
    print("\nPlain English Explanation:")
    print(f"The model predicted a match score of {prediction:.2f} for this candidate and job.")
    
    # Find top positive and top negative contributors
    contributions.sort(key=lambda x: x[2], reverse=True)
    top_pos = contributions[0]
    top_neg = contributions[-1]
    
    if top_pos[2] > 0:
        print(f"The biggest reason for increasing this score was the '{top_pos[0]}' (value: {top_pos[1]:.2f}), "
              f"which pushed the score up by {top_pos[2]:.2f}.")
    if top_neg[2] < 0:
        print(f"On the other hand, the score was brought down mostly by '{top_neg[0]}' (value: {top_neg[1]:.2f}), "
              f"which reduced the score by {abs(top_neg[2]):.2f}.")
              
if __name__ == "__main__":
    explain_prediction()
