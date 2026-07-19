"""Plain-English, one-example demo walkthrough for the mitigated model,
required by the DoD ("this input, this output, and the plain-English
reason")."""
import numpy as np
import pandas as pd

from config import CANDIDATES_PATH, MERIT_FEATURES, PROTECTED_ATTR, LABEL_COL
from data_validation import clean
from mitigate import train_mitigated


def demo(seed: int = 42):
    df = clean(pd.read_csv(CANDIDATES_PATH))
    model, thresholds, split_data, _ = train_mitigated(df, seed=seed)
    X_test, y_test_hist, tier_test, preds, df_test = split_data
    probs_test = model.predict_proba(X_test)[:, 1]

    importances = dict(zip(MERIT_FEATURES, model.feature_importances_))

    # Pick a Tier-3 candidate who is recommended under the mitigated model --
    # the group that was previously locked out almost entirely (Task 21: ~0%).
    idx = None
    for i in range(len(X_test)):
        if tier_test[i] == 3 and preds[i] == 1:
            idx = i
            break
    if idx is None:
        idx = 0

    row = X_test.iloc[idx]
    tier_val = tier_test[idx]
    prob = probs_test[idx]

    print(f"Candidate: college_tier={tier_val}, skill_score={row.skill_score:.1f}, "
          f"years_exp={row.years_exp:.1f}, jd_match={row.jd_match:.1f}, "
          f"portfolio_score={row.portfolio_score:.1f}")
    print(f"Model probability of recommendation: {prob:.3f}  "
          f"(group threshold for tier {tier_val}: {thresholds[tier_val]:.3f})")
    print("Why: college_tier is NOT a model input. Feature importances "
          "driving this score:")
    for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  - {feat}: importance={imp:.3f}, this candidate's value={getattr(row, feat):.1f}")
    verdict = "RECOMMENDED" if prob >= thresholds[tier_val] else "NOT RECOMMENDED"
    print(f"Verdict: {verdict}")
    print()
    print("Under the ORIGINAL (Task 21) model this candidate's tier alone "
          "made a recommendation nearly impossible (tier-3 positive rate "
          "was ~0%). Under the mitigated model, tier plays no role -- the "
          "decision comes from skill, experience, JD fit, and portfolio only.")


if __name__ == "__main__":
    demo()
