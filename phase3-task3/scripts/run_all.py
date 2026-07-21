"""
Stage E — Integrate, break it, then demo.
Runs the full pipeline end to end on real (generated, frozen) data:
  A. data generation
  B. baseline model + latency profile (BEFORE)
  C. train/serve skew check, then optimized model + serving path (AFTER)
  D. before/after latency + cost numbers
  E. fairness sanity check, failure injection, worked example, final report

Run with:  python3 -m scripts.run_all   (from the project root)
"""
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (INTERACTIONS_CSV, EXP_DIR, REPORT_DIR, ALL_FEATURES,
                         LABEL_COL, LATENCY_P95_SLO_MS, REQUESTS_PER_DAY)
from data.generate_data import generate
from src import train_baseline, train_optimized, skew_check
from src.profiling import run_before, run_after, cost_report
from src.fairness_check import parity_check
from src.utils import ModelRegistry, log_experiment


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("STAGE A - generate real (frozen, reproducible) interaction log")
    df = generate()
    print(f"{len(df)} rows across {df.job_id.nunique()} jobs written to {INTERACTIONS_CSV}")

    section("STAGE B - train baseline model + evaluate on held-out test")
    baseline_model, baseline_metrics, baseline_params = train_baseline.main()

    section("Train/serve skew check (before trusting any offline number)")
    skew_check.main()

    section("STAGE B - latency profile of the BEFORE inference path")
    test_df = pd.read_csv(INTERACTIONS_CSV.replace(".csv", "_test.csv"))
    before = run_before(baseline_model, test_df)
    print(json.dumps(before, indent=2))

    section("STAGE C - build optimized model (distillation) meeting the SLO")
    optimized_model, optimized_metrics, optimized_params = train_optimized.main()

    section("STAGE C/D - latency profile of the AFTER inference path")
    after = run_after(optimized_model, test_df)
    print(json.dumps(after, indent=2))

    section("STAGE D - before/after cost estimate")
    cost_before = cost_report(before["latency"])
    cost_after = cost_report(after["latency_warm_cache"])
    print("BEFORE:", cost_before)
    print("AFTER: ", cost_after)

    section("Fairness sanity check (baseline vs optimized, not just at the end)")
    test_df_b = test_df.copy()
    test_df_b["score"] = baseline_model.predict(test_df_b[ALL_FEATURES].values)
    fairness_baseline = parity_check(test_df_b, "score")

    test_df_o = test_df.copy()
    test_df_o["score"] = optimized_model.predict(test_df_o[ALL_FEATURES].values)
    fairness_optimized = parity_check(test_df_o, "score")
    print("Baseline fairness:", fairness_baseline)
    print("Optimized fairness:", fairness_optimized)

    section("Stage E - deliberately induce failure, confirm designed degradation")
    from src.serving import OptimizedServer, LRUScoreCache, popularity_fallback_scorer
    from src.data_pipeline import FeatureStore
    fs_fail = FeatureStore(test_df)
    failing_server = OptimizedServer(optimized_model, fs_fail, cache=LRUScoreCache(),
                                      fallback_scorer=popularity_fallback_scorer,
                                      force_unavailable=True)
    probe_job = test_df.job_id.iloc[0]
    probe_rows = test_df[test_df.job_id == probe_job].to_dict(orient="records")
    fail_scores, mode_used = failing_server.rank(
        probe_job, probe_rows, cache_key_fn=lambda r: r["candidate_id"])
    print(f"Model forced unavailable -> serving path used '{mode_used}' path "
          f"(expected 'fallback'); request still returned {len(fail_scores)} scores, "
          f"no 5xx.")
    assert mode_used == "fallback", "designed degradation did not trigger!"
    failure_demo = {"forced_unavailable": True, "path_used": mode_used,
                     "n_scores_returned": int(len(fail_scores)),
                     "request_failed": False}

    section("Worked example (explainability)")
    example_job = test_df.job_id.iloc[0]
    example_rows = test_df[test_df.job_id == example_job].copy()
    example_rows["score"] = optimized_model.predict(example_rows[ALL_FEATURES].values)
    top = example_rows.sort_values("score", ascending=False).iloc[0]
    importances = dict(zip(ALL_FEATURES, optimized_model.feature_importances_))
    top_feature = max(importances, key=importances.get)

    worked_example = {
        "job_id": example_job,
        "top_candidate": top["candidate_id"],
        "predicted_score": float(top["score"]),
        "true_relevance_label": int(top[LABEL_COL]),
        "plain_english_reason": (
            f"Ranked #1 mainly due to '{top_feature}' "
            f"(value={top[top_feature]:.2f}, global feature importance="
            f"{importances[top_feature]:.2f}), combined with "
            f"skill_overlap={top['skill_overlap']:.2f} and "
            f"years_experience={top['years_experience']:.1f}."
        ),
        "feature_importances_optimized_model": {k: round(v, 4) for k, v in importances.items()},
    }
    print(json.dumps(worked_example, indent=2))

    section("FINAL SUMMARY")
    summary = {
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "quality_delta": {k: round(optimized_metrics[k] - baseline_metrics[k], 4)
                           for k in baseline_metrics if k != "n_queries"},
        "before_latency": before["latency"],
        "after_latency_cold": after["latency_cold_cache"],
        "after_latency_warm": after["latency_warm_cache"],
        "before_meets_slo": before["meets_slo"],
        "after_meets_slo": after["meets_slo"],
        "slo_ms": LATENCY_P95_SLO_MS,
        "p95_speedup_x": round(before["latency"]["p95_ms"] / after["latency_warm_cache"]["p95_ms"], 2),
        "cost_before": cost_before,
        "cost_after": cost_after,
        "daily_cost_savings_usd": round(cost_before["estimated_daily_cost_usd"]
                                         - cost_after["estimated_daily_cost_usd"], 2),
        "model_size_reduction_pct": round(
            100 * (1 - optimized_params["total_nodes"] / baseline_params["total_nodes"]), 1),
        "fairness_baseline": fairness_baseline,
        "fairness_optimized": fairness_optimized,
        "failure_injection_demo": failure_demo,
        "worked_example": worked_example,
    }
    with open(f"{EXP_DIR}/metrics_before_after.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    log_experiment({"stage": "run_all_summary", "summary": summary})
    return summary


if __name__ == "__main__":
    main()
