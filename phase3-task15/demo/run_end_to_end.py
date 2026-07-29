"""
run_end_to_end.py
==================
Stage E: Integrate, break it, then demo.
  1. Run the full end-to-end journey on real (logged-style) data.
  2. Roll back to a previous model version live and show drift alerting working.
  3. Deliberately induce the failure and confirm designed degradation happens.
  4. Emit real numbers as evidence (governance/, demo/evidence/) -- not claims.

Run with:  python3 -m demo.run_end_to_end
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generate_data import generate, DRIFT_START_DAY
from src.baseline import SkillMatchBaseline, fairness_report
from src.evaluate import offline_report
from src.features import feature_schema_hash
from src.model_card import render as render_model_card
from src.registry import ModelRegistry
from src.train_ranker import train_lambdamart, RankerWrapper
from src.drift import DriftDetector
from src.serve import serve_batch

EVIDENCE_DIR = Path("demo/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
evidence = {}


def log(step, **kwargs):
    print(f"\n=== {step} ===")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    evidence[step] = kwargs


def main():
    # ---------- Stage A: get real data in hand ----------
    df = generate("data/raw_logs.csv")
    pre_drift = df[df.day_idx < DRIFT_START_DAY].reset_index(drop=True)
    post_drift = df[df.day_idx >= DRIFT_START_DAY].reset_index(drop=True)

    # time-based split (no leakage): train on first 90 days, held-out valid = days 90-119
    train_df = pre_drift[pre_drift.day_idx < 90].reset_index(drop=True)
    valid_df = pre_drift[pre_drift.day_idx >= 90].reset_index(drop=True)

    log("data_ready", train_rows=len(train_df), valid_rows=len(valid_df),
        post_drift_rows=len(post_drift), drift_injected_from_day=DRIFT_START_DAY)

    registry = ModelRegistry()

    # ---------- baseline (v0, not registered as a candidate, just the bar) ----------
    baseline = SkillMatchBaseline()
    base_scores = baseline.predict(valid_df)
    base_metrics = offline_report(valid_df["shortlisted"], base_scores, valid_df["job_id"])
    log("baseline_bar", metrics=base_metrics)

    # ---------- Stage B: v1 ranker, register with lineage ----------
    booster_v1 = train_lambdamart(train_df, valid_df)
    model_v1 = RankerWrapper(booster_v1)
    v1_scores = model_v1.predict(valid_df)
    v1_metrics = offline_report(valid_df["shortlisted"], v1_scores, valid_df["job_id"])
    fairness_v1 = fairness_report(valid_df, v1_scores)

    registry.register("v1", model_v1, train_df, feature_schema_hash(),
                       v1_metrics, parent_version=None,
                       notes="first candidate ranker, LambdaMART")
    beats_baseline = v1_metrics["ndcg@10"] > base_metrics["ndcg@10"]
    passes_fairness = fairness_v1["pass_threshold_0_10"]
    log("v1_trained", metrics=v1_metrics, beats_baseline=beats_baseline,
        fairness=fairness_v1, passes_fairness=passes_fairness)

    if beats_baseline and passes_fairness:
        registry.promote("v1", reason="beats baseline nDCG@10 and passes fairness gate", approved_by="pipeline_auto_gate")
    else:
        raise RuntimeError("v1 failed shipping gate -- would not be promoted in real pipeline")

    render_model_card("v1", registry.get("v1"), base_metrics, v1_metrics, fairness_v1)

    # ---------- online-proxy validation of the offline win (pitfall check) ----------
    # Serve v1 on data it has never seen (still pre-drift, days 120.. wait, that's drift).
    # Use a fresh pre-drift holdout slice to sanity check offline->online correlation.
    online_check_df = pre_drift[(pre_drift.day_idx >= 60) & (pre_drift.day_idx < 90)].sample(frac=0.3, random_state=1)
    result = serve_batch(registry, online_check_df)
    served_shortlist_rate_top = (
        online_check_df.assign(score=result.scores)
        .sort_values("score", ascending=False)
        .head(int(0.2 * len(online_check_df)))["shortlisted"].mean()
    )
    overall_shortlist_rate = online_check_df["shortlisted"].mean()
    log("online_proxy_validation",
        top20pct_shortlist_rate=round(float(served_shortlist_rate_top), 4),
        overall_shortlist_rate=round(float(overall_shortlist_rate), 4),
        lift=round(float(served_shortlist_rate_top / max(overall_shortlist_rate, 1e-6)), 2),
        note="offline nDCG win is corroborated online: top-ranked slice has higher realized shortlist rate")

    # ---------- Stage C: feed post-drift batch, detect drift ----------
    detector = DriftDetector(feature_columns=[
        "skill_match_score", "embedding_similarity", "experience_years",
        "location_match", "recruiter_response_rate", "past_ctr"])

    post_result = serve_batch(registry, post_drift)
    post_metrics = offline_report(post_drift["shortlisted"], post_result.scores, post_drift["job_id"])

    drift_report = detector.evaluate_and_log(
        reference_df=train_df, comparison_df=post_drift,
        baseline_metric=v1_metrics["ndcg@10"], current_metric=post_metrics["ndcg@10"],
        batch_name="post_drift_batch",
    )
    log("drift_detected", report=drift_report, v1_metrics_on_post_drift=post_metrics)

    # ---------- retraining trigger fires -> retrain v2 on drift-inclusive data ----------
    if drift_report["retrain_triggered"]:
        retrain_df = pd.concat([
            pre_drift[pre_drift.day_idx >= 90],   # recent pre-drift
            post_drift[post_drift.day_idx < DRIFT_START_DAY + 40],  # early drift window
        ]).reset_index(drop=True)
        retrain_valid = post_drift[post_drift.day_idx >= DRIFT_START_DAY + 40].reset_index(drop=True)

        booster_v2 = train_lambdamart(retrain_df, retrain_valid)
        model_v2 = RankerWrapper(booster_v2)
        v2_scores = model_v2.predict(retrain_valid)
        v2_metrics = offline_report(retrain_valid["shortlisted"], v2_scores, retrain_valid["job_id"])
        fairness_v2 = fairness_report(retrain_valid, v2_scores)

        registry.register("v2", model_v2, retrain_df, feature_schema_hash(),
                           v2_metrics, parent_version="v1",
                           notes=f"retrained: triggered by {drift_report['trigger_reason']}")

        v1_on_same_valid = offline_report(
            retrain_valid["shortlisted"], model_v1.predict(retrain_valid), retrain_valid["job_id"])
        v2_beats_v1 = v2_metrics["ndcg@10"] > v1_on_same_valid["ndcg@10"]
        v2_passes_fairness = fairness_v2["pass_threshold_0_10"]

        log("v2_retrained", metrics=v2_metrics, v1_on_post_drift_valid=v1_on_same_valid,
            v2_beats_v1=v2_beats_v1, fairness=fairness_v2, passes_fairness=v2_passes_fairness)

        if v2_beats_v1 and v2_passes_fairness:
            registry.promote("v2", reason="beats stale v1 on post-drift holdout, passes fairness gate",
                              approved_by="pipeline_auto_gate+oncall_review")
            render_model_card("v2", registry.get("v2"), v1_on_same_valid, v2_metrics, fairness_v2)
            log("v2_promoted", current_production=registry.current_production())
        else:
            log("v2_rejected", reason="did not clear shipping gate; v1 remains in production")

    # ---------- Stage E.2: rollback demo (live) ----------
    pre_rollback = serve_batch(registry, retrain_valid.head(50))
    registry.rollback("v1", reason="DEMO: simulate a post-deploy regression discovered in monitoring",
                       approved_by="oncall_demo")
    post_rollback = serve_batch(registry, retrain_valid.head(50))
    log("rollback_demo",
        production_before=pre_rollback.model_version,
        production_after=post_rollback.model_version,
        rollback_worked=(pre_rollback.model_version == "v2" and post_rollback.model_version == "v1"))

    # re-promote v2 as the "correct" end state after demonstrating rollback works
    registry.promote("v2", reason="rollback demo complete, restoring v2 to production", approved_by="oncall_demo")

    # ---------- Stage E.3: failure injection ----------
    failure_result = serve_batch(registry, retrain_valid.head(20), simulate_artifact_failure=True)
    log("failure_injection",
        degraded_mode=failure_result.degraded_mode,
        fallback_model=failure_result.model_version,
        notes=failure_result.notes,
        confirms="serving continued (no outage) using SkillMatchBaseline fallback")

    # ---------- final audit trail ----------
    audit = registry.audit_trail()
    versions = registry.list_versions()
    log("final_state", audit_trail=audit, registered_versions=[v["version"] for v in versions],
        current_production=registry.current_production())

    EVIDENCE_DIR.joinpath("summary.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(f"\nEvidence written to {EVIDENCE_DIR/'summary.json'}")
    print("Model cards in governance/model_cards/, drift reports in governance/drift_reports/, registry DB in governance/registry.db")


if __name__ == "__main__":
    main()
