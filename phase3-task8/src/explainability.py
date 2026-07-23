"""
explainability.py
------------------
Stage B.4: "Produce one worked example: this input, this output, this
plain-English reason -- plus what happens when the model is unavailable."

We don't have SHAP available offline (no network to install it), so we use
two honest, dependency-free techniques instead, and say so:
  1. GLOBAL importance: sklearn's permutation_importance on the holdout set
     (model-agnostic, no extra dependency, directly answers "which features
     actually move holdout performance").
  2. PER-CANDIDATE reason codes: for a given candidate, compare each feature
     value to the training population's mean/std (z-score), then surface the
     top-3 features with the largest |z-score| among the features that
     ALSO rank high in global importance -- i.e. "what's most unusual about
     this person, restricted to what the model actually listens to."
     This is an approximation of SHAP, not a replacement -- documented here
     rather than silently claimed as SHAP.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]

READABLE_NAMES = {
    "days_since_signup": "account age",
    "days_since_last_event": "days since last activity",
    "events_7d": "activity in the last 7 days",
    "events_14d": "activity in the last 14 days",
    "events_30d": "activity in the last 30 days",
    "events_90d": "activity in the last 90 days",
    "apply_30d": "job applications in the last 30 days",
    "apply_90d": "job applications in the last 90 days",
    "login_30d": "logins in the last 30 days",
    "distinct_event_types_30d": "variety of actions taken recently",
    "recency_trend_ratio": "whether activity is speeding up or slowing down",
    "seniority_enc": "seniority level",
    "city_tier_enc": "city tier",
    "channel_enc": "acquisition channel",
    "device_enc": "device type",
}


def compute_global_importance():
    holdout = pd.read_csv(ROOT / "data/processed/holdout_snapshots.csv")
    X, y = holdout[FEATURE_COLUMNS], holdout["churned"]
    with open(ROOT / "models/churn_model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    r = permutation_importance(model, X, y, n_repeats=8, random_state=42, scoring="average_precision")
    imp = pd.Series(r.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
    imp.to_csv(ROOT / "outputs/global_feature_importance.csv", header=["importance"])
    return imp


def reason_codes_for_row(row, pop_mean, pop_std, importance, top_n=3):
    z = {}
    for c in FEATURE_COLUMNS:
        std = pop_std[c] if pop_std[c] > 1e-9 else 1.0
        z[c] = (row[c] - pop_mean[c]) / std
    ranked = sorted(FEATURE_COLUMNS, key=lambda c: importance.get(c, 0) * abs(z[c]), reverse=True)
    reasons = []
    for c in ranked[:top_n]:
        direction = "higher than typical" if z[c] > 0 else "lower than typical"
        reasons.append(f"{READABLE_NAMES.get(c, c)} is {direction} (z={z[c]:.1f})")
    return reasons


def build_worked_example():
    holdout = pd.read_csv(ROOT / "data/processed/holdout_snapshots.csv")
    train = pd.read_csv(ROOT / "data/processed/train_snapshots.csv")
    with open(ROOT / "models/churn_model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    importance = compute_global_importance().to_dict()

    pop_mean = train[FEATURE_COLUMNS].mean()
    pop_std = train[FEATURE_COLUMNS].std()

    scores = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    holdout = holdout.assign(risk_score=scores)
    # pick one true-positive-looking example (high score AND actually churned) for an honest worked example
    candidates_tp = holdout[(holdout.risk_score > 0.6) & (holdout.churned == 1)]
    example = (candidates_tp.sample(1, random_state=7) if len(candidates_tp) else holdout.sample(1, random_state=7)).iloc[0]

    reasons = reason_codes_for_row(example, pop_mean, pop_std, importance)

    doc = {
        "candidate_id": example.candidate_id,
        "as_of_date": str(example.as_of_date),
        "input_features": {c: (float(example[c]) if isinstance(example[c], (int, float, np.floating, np.integer))
                                else example[c]) for c in FEATURE_COLUMNS},
        "model_output_risk_score": float(example.risk_score),
        "actual_outcome_next_21_days": "churned (no activity)" if example.churned == 1 else "stayed active",
        "plain_english_reasons": reasons,
        "model_unavailable_fallback": "If the model service is down, this candidate would instead be scored by "
                                       "the 14-day-inactivity rule using days_since_last_event only -- degraded "
                                       "but never silent (see failure_simulation.py).",
    }
    with open(ROOT / "outputs/worked_example.json", "w") as f:
        json.dump(doc, f, indent=2, default=str)

    md = [f"# Worked Example — Candidate {example.candidate_id}\n",
          f"**As-of date:** {example.as_of_date}",
          f"**Model risk score:** {example.risk_score:.3f}",
          f"**Actual outcome (next {21} days):** {doc['actual_outcome_next_21_days']}\n",
          "**Why the model flagged this candidate (plain English):**"]
    for r in reasons:
        md.append(f"- {r}")
    md.append(f"\n**If the model is unavailable:** {doc['model_unavailable_fallback']}")
    with open(ROOT / "outputs/worked_example.md", "w") as f:
        f.write("\n".join(md))

    print("[explainability] wrote outputs/worked_example.md, outputs/worked_example.json, "
          "outputs/global_feature_importance.csv")
    return doc


if __name__ == "__main__":
    build_worked_example()
