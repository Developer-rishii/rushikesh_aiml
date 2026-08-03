"""
Stage B.4/C.4/D.4 — "Make it explainable, safe & demoable... plus what
happens when the model is unavailable."

serve_ranking() is the single production entry point: model score ->
tenant policy -> guardrail-gated commit path -> explainable output.
If the model fails to load/predict, we fall back to the tenant's
guardrail-approved policy score computed on raw sub-signals only
(skill_overlap/experience/distance), degraded but never silently broken,
and we say so in the response ('degraded_mode': True) rather than pretending
nothing happened.
"""
import numpy as np
import pandas as pd
import joblib

from features import compute_features, FEATURE_COLUMNS
from policy import apply_policy, PolicyStore
from config import DATA_DIR
MODEL_PATH = DATA_DIR / "model.joblib"

def _safe_model_score(df):
    """Returns (scores, degraded_mode)."""
    try:
        model = joblib.load(MODEL_PATH)
        feats = compute_features(df)
        return model.predict(feats[FEATURE_COLUMNS]), False
    except Exception as e:  # model missing/corrupt/serving error
        print(f"[serve] MODEL UNAVAILABLE ({e}); falling back to rule-only score")
        feats = compute_features(df)
        fallback = 0.6 * feats["skill_overlap"] + 0.4 / (1 + np.exp(-0.2 * feats["exp_gap"]))
        return fallback.values, True


def explain_top_pick(row, config):
    """One plain-English reason per Stage requirement 'this input, this
    output, this plain-English reason'."""
    return (
        f"Candidate {row.candidate_id} ranked #1 for {row.job_id} because "
        f"skill overlap was {row.skill_overlap:.0%} (weight {config.w_skill}), "
        f"experience gap {row.years_exp - row.req_years_exp:+.1f} yrs "
        f"(weight {config.w_experience}), and distance {row.distance_km:.0f}km "
        f"(weight {config.w_distance}); combined policy_score={row.policy_score:.3f}."
    )


def serve_ranking(store: PolicyStore, tenant_id, job_id, candidates_df,
                   simulate_model_down=False):
    scores, degraded = (np.zeros(len(candidates_df)) - 1, True) if simulate_model_down \
        else _safe_model_score(candidates_df)
    df = candidates_df.copy()
    df["score"] = scores if not simulate_model_down else _safe_model_score(candidates_df)[0]
    # when simulating an outage we still want the fallback path exercised:
    if simulate_model_down:
        feats = compute_features(df)
        df["score"] = (0.6 * feats["skill_overlap"] +
                        0.4 / (1 + np.exp(-0.2 * feats["exp_gap"]))).values
        degraded = True

    config = store.get(tenant_id)
    ranked = apply_policy(df, config, base_score_col="score")
    ranked = ranked[ranked.eligible].sort_values("policy_score", ascending=False)

    explanation = None
    if len(ranked):
        explanation = explain_top_pick(ranked.iloc[0], config)

    return {
        "tenant_id": tenant_id, "job_id": job_id,
        "degraded_mode": degraded,
        "config_version": config.version,
        "top10": ranked[["candidate_id", "policy_score"]].head(10),
        "explanation": explanation,
        "n_eligible": int(ranked.shape[0]),
        "n_total": int(len(df)),
    }
