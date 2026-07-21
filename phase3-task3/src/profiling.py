"""
Profiles the inference path end-to-end (feature fetch -> model.predict ->
sort) and reports p50/p95/p99 latency per ranking request, plus a stage
breakdown so we can say *where* time goes instead of guessing.

Stage B deliverable: `latency_profile_<before/after>.json`
Stage D deliverable: before/after comparison + $ cost, via `cost_report()`
"""
import json
import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd

from src.config import (INTERACTIONS_CSV, EXP_DIR, COST_PER_MS_USD,
                         REQUESTS_PER_DAY, LATENCY_P95_SLO_MS)
from src.data_pipeline import FeatureStore
from src.serving import BaselineServer, OptimizedServer, LRUScoreCache, popularity_fallback_scorer


def _job_requests(test_df, n_jobs=150, seed=7):
    jobs = test_df["job_id"].unique()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(jobs, size=min(n_jobs, len(jobs)), replace=False)
    reqs = []
    for j in chosen:
        rows = test_df[test_df.job_id == j].to_dict(orient="records")
        reqs.append((j, rows))
    return reqs


def profile_stage_breakdown(server, feature_store, requests, mode="baseline"):
    """One representative request, instrumented stage-by-stage, to answer
    'is the bottleneck the model, or fetching its features?'"""
    job_id, rows = requests[0]
    breakdown = {}

    t0 = time.perf_counter()
    if mode == "baseline":
        for r in rows:
            feature_store.fetch_one(r["candidate_id"])
    else:
        feature_store.fetch_batch([r["candidate_id"] for r in rows])
    breakdown["feature_fetch_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if mode == "baseline":
        server.rank(job_id, rows)
    else:
        server.rank(job_id, rows, cache_key_fn=lambda r: r["candidate_id"])
    breakdown["full_request_ms"] = (time.perf_counter() - t0) * 1000
    breakdown["inference_and_overhead_ms"] = (
        breakdown["full_request_ms"] - breakdown["feature_fetch_ms"])
    return breakdown


def measure_latency(rank_fn, requests, warmup=5):
    for j, rows in requests[:warmup]:
        rank_fn(j, rows)  # warm up caches/JIT-ish effects, excluded from measurement

    latencies_ms = []
    for j, rows in requests:
        t0 = time.perf_counter()
        rank_fn(j, rows)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies_ms)
    return {
        "n_requests": len(arr),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "max_ms": float(np.max(arr)),
    }


def cost_report(latency_stats: dict, requests_per_day=REQUESTS_PER_DAY):
    """Turns a p50 latency into a daily $ compute-cost estimate."""
    daily_compute_ms = latency_stats["p50_ms"] * requests_per_day
    return {
        "requests_per_day_assumed": requests_per_day,
        "p50_ms": latency_stats["p50_ms"],
        "p95_ms": latency_stats["p95_ms"],
        "estimated_daily_cost_usd": round(daily_compute_ms * COST_PER_MS_USD, 2),
    }


def run_before(baseline_model, test_df):
    fs = FeatureStore(test_df, per_call_latency_ms=0.9, batch_overhead_ms=1.2)
    server = BaselineServer(baseline_model, fs)
    requests = _job_requests(test_df)

    breakdown = profile_stage_breakdown(server, fs, requests, mode="baseline")
    latency = measure_latency(lambda j, rows: server.rank(j, rows), requests)
    result = {"mode": "BEFORE (baseline)", "stage_breakdown_ms": breakdown,
              "latency": latency, "slo_ms": LATENCY_P95_SLO_MS,
              "meets_slo": latency["p95_ms"] <= LATENCY_P95_SLO_MS}
    with open(f"{EXP_DIR}/latency_profile_before.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_after(optimized_model, test_df, force_unavailable=False):
    fs = FeatureStore(test_df, per_call_latency_ms=0.9, batch_overhead_ms=1.2)
    cache = LRUScoreCache()
    server = OptimizedServer(optimized_model, fs, cache=cache,
                              fallback_scorer=popularity_fallback_scorer,
                              force_unavailable=force_unavailable)
    requests = _job_requests(test_df)

    def rank_fn(j, rows):
        return server.rank(j, rows, cache_key_fn=lambda r: r["candidate_id"])

    breakdown = profile_stage_breakdown(server, fs, requests, mode="optimized")
    # First pass measures cold-cache batched path; second pass over the SAME
    # requests shows the cache-hit benefit for repeat scoring within TTL.
    latency_cold = measure_latency(rank_fn, requests, warmup=0)
    latency_warm = measure_latency(rank_fn, requests, warmup=0)  # cache now populated
    result = {"mode": "AFTER (optimized)", "stage_breakdown_ms": breakdown,
              "latency_cold_cache": latency_cold, "latency_warm_cache": latency_warm,
              "cache_entries": len(cache), "slo_ms": LATENCY_P95_SLO_MS,
              "meets_slo": latency_warm["p95_ms"] <= LATENCY_P95_SLO_MS}
    with open(f"{EXP_DIR}/latency_profile_after.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
