"""
latency_bench.py — Stage C: latency results for the tenant, and
Stage B step 4 / Stage E step 3: "what happens when the model is
unavailable" / "deliberately induce the failure and confirm the
designed degradation actually happens."

Measures p50/p95/p99 latency for scoring one job's full candidate pool
(realistic serving unit: rank all applicants for a single req), using
the SAME build_features() path as training (train/serve skew guard).

Also runs a chaos test: simulate the model artifact being unavailable
and confirm the service falls back to the skill_overlap baseline
ranker instead of crashing or returning nothing.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import joblib

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS


def score_job(model, cands, job_row, jobs_df):
    single_job_df = pd.DataFrame([job_row])
    imp = pd.DataFrame({
        "candidate_id": cands["candidate_id"],
        "job_id": job_row["job_id"],
        "experience_years": cands["experience_years"],
        "clicked": 0, "shortlisted": 0, "hired": 0,
    })
    feats = build_features(imp, cands, jobs_df)
    if model is not None:
        return model.predict(feats[FEATURE_COLUMNS])
    # DEGRADED MODE: model unavailable -> fall back to transparent
    # skill_overlap ranking rather than failing the request.
    return feats["skill_overlap"].values


def main():
    cands = pd.read_csv(f"{DATA}/candidates.csv")
    jobs = pd.read_csv(f"{DATA}/jobs.csv")
    model = joblib.load(f"{MODELS}/ranker_v1.joblib")

    # --- Live latency: model available ---
    timings = []
    for _, job_row in jobs.iterrows():
        t0 = time.perf_counter()
        _ = score_job(model, cands, job_row, jobs)
        timings.append((time.perf_counter() - t0) * 1000)  # ms
    timings = np.array(timings)

    # --- Chaos test: model artifact "unavailable" ---
    chaos_ok = True
    chaos_error = None
    try:
        fallback_scores = score_job(None, cands, jobs.iloc[0], jobs)
        chaos_ok = len(fallback_scores) == len(cands) and not np.any(np.isnan(fallback_scores))
    except Exception as e:
        chaos_ok = False
        chaos_error = str(e)

    report = {
        "tenant": "AcmeFinServ_Pilot",
        "unit": "score full candidate pool (2000) for 1 job requisition",
        "n_jobs_benchmarked": len(jobs),
        "latency_ms": {
            "p50": round(float(np.percentile(timings, 50)), 2),
            "p95": round(float(np.percentile(timings, 95)), 2),
            "p99": round(float(np.percentile(timings, 99)), 2),
            "max": round(float(np.max(timings)), 2),
        },
        "chaos_test_model_unavailable": {
            "designed_degradation": "fall back to skill_overlap baseline ranking",
            "degradation_worked": chaos_ok,
            "error": chaos_error,
        },
    }
    with open(f"{EXPER}/latency_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
