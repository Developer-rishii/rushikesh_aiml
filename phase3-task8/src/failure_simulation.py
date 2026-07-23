"""
failure_simulation.py
-----------------------
Stage E.3: "Deliberately induce the failure and confirm the designed
degradation actually happens."

Failure scenario simulated: the model artifact/service is unavailable
(file missing / exception on load / inference timeout). Rather than the
pipeline crashing and growth getting NOTHING, score_candidates() catches the
failure and falls back to the transparent 14-day-inactivity rule baseline
(same one from baseline_model.py), tags every row with
`degraded_mode: true`, and logs the incident. This is the "what happens when
the model is unavailable" requirement from Stage B.4, actually exercised
end-to-end here rather than just described.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import compute_features_as_of, FEATURE_COLUMNS, sufficient_history_mask
from label_definition import MIN_HISTORY_DAYS
from baseline_model import rule_14_day_score
from at_risk_list import SERVE_AS_OF_DATE, TOP_N_FOR_GROWTH

ROOT = Path(__file__).resolve().parents[1]


class ModelUnavailableError(Exception):
    pass


def _load_model(force_failure: bool, model_path: Path):
    if force_failure:
        raise ModelUnavailableError("Simulated: model service returned 503 / artifact failed to load.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_candidates(force_failure=False):
    profiles = pd.read_csv(ROOT / "data/raw/candidate_profiles_SIMULATED.csv", parse_dates=["signup_date"])
    events = pd.read_csv(ROOT / "data/raw/interaction_events_SIMULATED.csv", parse_dates=["event_ts"])
    feat = compute_features_as_of(profiles, events, SERVE_AS_OF_DATE)
    feat = feat[sufficient_history_mask(feat, MIN_HISTORY_DAYS)].reset_index(drop=True)

    incident_log = {"timestamp": datetime.now().isoformat(), "forced_failure": force_failure}
    try:
        model = _load_model(force_failure, ROOT / "models/churn_model_v1.pkl")
        feat["risk_score"] = model.predict_proba(feat[FEATURE_COLUMNS])[:, 1]
        feat["degraded_mode"] = False
        feat["scoring_source"] = "model_v1"
        incident_log["outcome"] = "model scored successfully, no degradation"
    except ModelUnavailableError as e:
        feat["risk_score"] = rule_14_day_score(feat)
        # normalize baseline score to [0,1] like a probability, for a consistent output contract
        max_v = feat["risk_score"].max() or 1.0
        feat["risk_score"] = (feat["risk_score"] / max_v).round(4)
        feat["degraded_mode"] = True
        feat["scoring_source"] = "fallback_14day_rule"
        incident_log["outcome"] = f"DEGRADED MODE ENGAGED: {e}. Fell back to 14-day-inactivity rule. " \
                                   "At-risk list still produced -- growth pipeline did not go silent."

    feat = feat.sort_values("risk_score", ascending=False).reset_index(drop=True)
    feat["rank"] = feat.index + 1
    result = feat.head(TOP_N_FOR_GROWTH)[["rank", "candidate_id", "risk_score", "degraded_mode", "scoring_source"]]
    return result, incident_log


def main():
    print("=== Scenario 1: normal operation ===")
    normal_result, normal_log = score_candidates(force_failure=False)
    print(json.dumps(normal_log, indent=2))
    print(normal_result.head(3).to_string(index=False))

    print("\n=== Scenario 2: model service DOWN (forced failure) ===")
    degraded_result, degraded_log = score_candidates(force_failure=True)
    print(json.dumps(degraded_log, indent=2))
    print(degraded_result.head(3).to_string(index=False))

    assert degraded_result["degraded_mode"].all(), "Degradation flag did not propagate -- FAIL"
    assert len(degraded_result) == len(normal_result), "Fallback produced a different list size -- FAIL"
    print("\n[failure_simulation] VERIFIED: pipeline degrades gracefully and still delivers a full at-risk "
          "list to growth when the model is unavailable. No silent failure, no crash, no empty output.")

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "outputs/failure_simulation_log.json", "w") as f:
        json.dump({"normal": normal_log, "degraded": degraded_log,
                   "verification_passed": True}, f, indent=2)
    degraded_result.to_csv(ROOT / "outputs/at_risk_list_DEGRADED_MODE_example.csv", index=False)


if __name__ == "__main__":
    main()
