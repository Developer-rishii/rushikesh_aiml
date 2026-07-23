"""
at_risk_list.py
-----------------
Stage D: "A prioritised at-risk list handed to growth".

Uses the SAME compute_features_as_of() function as training (no train/serve
skew) to score candidates as of the latest date we have data for, then
outputs a ranked CSV that Growth can act on directly: candidate_id, risk
score, rank, reason codes, and a suggested lever (per the "actionability"
core concept -- a score with no lever is a vanity model).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import compute_features_as_of, FEATURE_COLUMNS, sufficient_history_mask
from label_definition import MIN_HISTORY_DAYS
from explainability import reason_codes_for_row, compute_global_importance

ROOT = Path(__file__).resolve().parents[1]
SERVE_AS_OF_DATE = pd.Timestamp("2026-01-10")  # "today" for serving purposes -- latest date we have logs for
TOP_N_FOR_GROWTH = 500


def _lever_for(row):
    if row["apply_30d"] == 0 and row["events_30d"] > 0:
        return "Send curated job digest (browsing but not applying)"
    if row["login_30d"] <= 1:
        return "Re-engagement push notification / email"
    if row["recency_trend_ratio"] < 0.5:
        return "Personal outreach (activity decelerating fast)"
    return "Add to standard nurture campaign"


def main():
    profiles = pd.read_csv(ROOT / "data/raw/candidate_profiles_SIMULATED.csv", parse_dates=["signup_date"])
    events = pd.read_csv(ROOT / "data/raw/interaction_events_SIMULATED.csv", parse_dates=["event_ts"])
    train = pd.read_csv(ROOT / "data/processed/train_snapshots.csv")

    feat = compute_features_as_of(profiles, events, SERVE_AS_OF_DATE)
    feat = feat[sufficient_history_mask(feat, MIN_HISTORY_DAYS)].reset_index(drop=True)

    with open(ROOT / "models/churn_model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    feat["risk_score"] = model.predict_proba(feat[FEATURE_COLUMNS])[:, 1]

    importance = compute_global_importance().to_dict()
    pop_mean, pop_std = train[FEATURE_COLUMNS].mean(), train[FEATURE_COLUMNS].std()

    feat = feat.sort_values("risk_score", ascending=False).reset_index(drop=True)
    feat["rank"] = np.arange(1, len(feat) + 1)
    top = feat.head(TOP_N_FOR_GROWTH).copy()

    top["reason_codes"] = top.apply(
        lambda r: "; ".join(reason_codes_for_row(r, pop_mean, pop_std, importance)), axis=1)
    top["suggested_lever"] = top.apply(_lever_for, axis=1)

    out_cols = ["rank", "candidate_id", "risk_score", "days_since_last_event", "apply_30d",
                "events_30d", "reason_codes", "suggested_lever"]
    out = top[out_cols].copy()
    out["risk_score"] = out["risk_score"].round(4)
    out["as_of_date"] = SERVE_AS_OF_DATE.date()
    out["model_version"] = "churn_model_v1"

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "outputs/at_risk_list_for_growth.csv", index=False)

    summary = {
        "as_of_date": str(SERVE_AS_OF_DATE.date()),
        "n_scored_total": int(len(feat)),
        "n_handed_to_growth": int(len(out)),
        "mean_risk_score_top_list": float(out.risk_score.mean()),
        "lever_breakdown": out["suggested_lever"].value_counts().to_dict(),
    }
    with open(ROOT / "outputs/at_risk_list_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[at_risk_list] scored {len(feat)} eligible candidates, handed top {len(out)} to growth")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
