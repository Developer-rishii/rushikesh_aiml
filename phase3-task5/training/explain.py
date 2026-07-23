"""
Explainability script using SHAP TreeExplainer.
Generates plain-English explanations for multiple candidate-job pairs
and saves them to results/explain_example.md.
"""
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def explain_prediction(num_examples=5):
    root = get_project_root()
    df = pd.read_csv(os.path.join(root, 'data', 'interactions.csv'))
    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']

    unique_candidates = np.array(df['candidate_id'].unique())
    _, test_cands = train_test_split(unique_candidates, test_size=0.2, random_state=42)
    test_df = df[df['candidate_id'].isin(test_cands)].copy().reset_index(drop=True)

    X_test = test_df[features]

    model = lgb.Booster(model_file=os.path.join(root, 'models', 'lgbm_ranker.txt'))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[0]

    lines = []
    lines.append("# Explainability Report")
    lines.append("")
    lines.append(f"Generated programmatically by `training/explain.py`.")
    lines.append(f"Base Value (Average Prediction across all data): {base_value:.4f}")
    lines.append("")

    for idx in range(min(num_examples, len(test_df))):
        candidate = test_df.iloc[idx]['candidate_id']
        job = test_df.iloc[idx]['job_id']
        prediction = model.predict(X_test.iloc[[idx]])[0]

        lines.append(f"---")
        lines.append(f"## Example {idx + 1}: Candidate {candidate} -> Job {job}")
        lines.append(f"- **Predicted Score:** {prediction:.4f}")
        lines.append(f"")
        lines.append(f"### Feature Contributions")
        lines.append(f"| Feature | Value | SHAP Contribution |")
        lines.append(f"|---------|-------|--------------------|")

        contributions = []
        for i, feature in enumerate(features):
            val = X_test.iloc[idx][feature]
            contrib = shap_values[idx][i]
            contributions.append((feature, val, contrib))
            lines.append(f"| {feature} | {val:.2f} | {contrib:+.4f} |")

        lines.append(f"")

        # Sort for narrative
        contributions.sort(key=lambda x: x[2], reverse=True)
        top_pos = contributions[0]
        top_neg = contributions[-1]

        lines.append(f"### Plain English Explanation")
        explanation = f"The model predicted a match score of {prediction:.4f} for candidate {candidate} and job {job}."
        if top_pos[2] > 0:
            explanation += (
                f" The biggest reason for increasing this score was '{top_pos[0]}' "
                f"(value: {top_pos[1]:.2f}), which pushed the score up by {top_pos[2]:.4f}."
            )
        if top_neg[2] < 0:
            explanation += (
                f" On the other hand, the score was brought down mostly by '{top_neg[0]}' "
                f"(value: {top_neg[1]:.2f}), which reduced the score by {abs(top_neg[2]):.4f}."
            )
        lines.append(explanation)
        lines.append("")

    report = "\n".join(lines)
    print(report)

    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'explain_example.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    explain_prediction()
