"""
Fairness evaluation script.
Saves raw output to results/fairness_report.md so the sign-off doc can cite it.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os, sys

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def eval_fairness():
    root = get_project_root()
    print("Evaluating fairness on test set...")
    df = pd.read_csv(os.path.join(root, 'data', 'interactions.csv'))

    unique_candidates = np.array(df['candidate_id'].unique())
    _, test_cands = train_test_split(unique_candidates, test_size=0.2, random_state=42)
    test_df = df[df['candidate_id'].isin(test_cands)].copy()

    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
    X_test = test_df[features]

    model = lgb.Booster(model_file=os.path.join(root, 'models', 'lgbm_ranker.txt'))

    test_df['predicted_score'] = model.predict(X_test)

    group0 = test_df[test_df['demographic_group'] == 0]
    group1 = test_df[test_df['demographic_group'] == 1]

    dp0 = group0['predicted_score'].mean()
    dp1 = group1['predicted_score'].mean()
    dp_gap = dp0 - dp1

    rel_group0 = group0[group0['relevance'] > 0]
    rel_group1 = group1[group1['relevance'] > 0]

    eo0 = rel_group0['predicted_score'].mean()
    eo1 = rel_group1['predicted_score'].mean()
    eo_gap = eo0 - eo1

    lines = []
    lines.append("# Fairness Evaluation Report")
    lines.append(f"")
    lines.append(f"Generated programmatically by `training/eval_fairness.py`.")
    lines.append(f"")
    lines.append(f"## Demographic Parity Check")
    lines.append(f"- Group 0 (Majority) Mean Predicted Score: {dp0:.4f}")
    lines.append(f"- Group 1 (Minority) Mean Predicted Score: {dp1:.4f}")
    lines.append(f"- Gap (Group 0 - Group 1): {dp_gap:.4f}")
    lines.append(f"- Direction: {'Group 0 scored higher' if dp_gap > 0 else 'Group 1 scored higher'}")
    lines.append(f"")
    lines.append(f"## Equal Opportunity Check (among candidates with relevance > 0)")
    lines.append(f"- Group 0 Mean Predicted Score: {eo0:.4f}")
    lines.append(f"- Group 1 Mean Predicted Score: {eo1:.4f}")
    lines.append(f"- Gap (Group 0 - Group 1): {eo_gap:.4f}")
    lines.append(f"- Direction: {'Group 0 scored higher' if eo_gap > 0 else 'Group 1 scored higher'}")
    lines.append(f"")
    lines.append(f"## Interpretation")
    if dp_gap > 0 and eo_gap < 0:
        lines.append(
            f"The model shows a mixed fairness picture. On demographic parity (all candidates), "
            f"Group 0 scores slightly higher (gap={dp_gap:.4f}), suggesting the model has absorbed "
            f"some of the historical bias baked into the training data. However, on equal opportunity "
            f"(only truly relevant candidate-job pairs), Group 1 actually scores higher (gap={eo_gap:.4f}). "
            f"This means that among candidates who genuinely applied, the model is more generous "
            f"to Group 1 — possibly because Group 1 candidates who overcame the historical bias "
            f"to actually apply had stronger signals on other features (experience, skills). "
            f"This discrepancy is important: the demographic parity gap and the equal opportunity "
            f"gap point in opposite directions, so a single 'biased against Group 1' narrative "
            f"would be inaccurate."
        )
    elif dp_gap > 0 and eo_gap > 0:
        lines.append(
            f"Both metrics show Group 0 scoring higher. The model has absorbed historical bias "
            f"against Group 1 on both demographic parity (gap={dp_gap:.4f}) and equal opportunity "
            f"(gap={eo_gap:.4f}). Remediation is needed before production deployment."
        )
    else:
        lines.append(
            f"Demographic parity gap: {dp_gap:.4f}, Equal opportunity gap: {eo_gap:.4f}. "
            f"Review the signs and magnitudes above for a nuanced interpretation."
        )

    report = "\n".join(lines)

    # Print to stdout
    print(report)

    # Save to results/
    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'fairness_report.md')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    eval_fairness()
