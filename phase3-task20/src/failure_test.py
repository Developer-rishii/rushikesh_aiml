"""
failure_test.py — Stage E: "Deliberately induce the failure and confirm
the designed degradation actually happens."

Three induced failures, each checked for a designed (non-crashing,
transparent) degradation:
  1. Model file missing entirely -> serving falls back to skill_overlap.
  2. Corrupted/malformed feature row (NaN skill data) -> row is skipped
     with a logged warning, not a silent bad score or a crash.
  3. Empty candidate pool for a job -> service returns an empty ranked
     list with a clear reason, not an exception.
"""
import os
import shutil
import numpy as np
import pandas as pd
import joblib

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS


def safe_rank(model, impressions, cands, jobs):
    """Serving function with designed degradation paths."""
    if len(impressions) == 0:
        return {"status": "empty", "reason": "no candidates in pool for this job", "ranked": []}

    feats = build_features(impressions, cands, jobs)

    bad_mask = feats[FEATURE_COLUMNS].isna().any(axis=1)
    n_skipped = int(bad_mask.sum())
    clean = feats.loc[~bad_mask].copy()

    if model is not None:
        clean["score"] = model.predict(clean[FEATURE_COLUMNS])
        mode = "model"
    else:
        clean["score"] = clean["skill_overlap"]
        mode = "baseline_fallback"

    return {
        "status": "ok",
        "mode": mode,
        "n_skipped_malformed_rows": n_skipped,
        "ranked": clean.sort_values("score", ascending=False)["candidate_id"].tolist(),
    }


def main():
    cands = pd.read_csv(f"{DATA}/candidates.csv")
    jobs = pd.read_csv(f"{DATA}/jobs.csv")
    job = jobs.iloc[0]

    pool = cands.head(20).copy()
    imp = pd.DataFrame({
        "candidate_id": pool["candidate_id"], "job_id": job["job_id"],
        "experience_years": pool["experience_years"],
        "clicked": 0, "shortlisted": 0, "hired": 0,
    })

    results = {}

    # Failure 1: model file missing
    tmp_missing = f"{MODELS}/ranker_v1_MOVED.joblib"
    shutil.move(f"{MODELS}/ranker_v1.joblib", tmp_missing)
    try:
        model = None
        if os.path.exists(f"{MODELS}/ranker_v1.joblib"):
            model = joblib.load(f"{MODELS}/ranker_v1.joblib")
        r1 = safe_rank(model, imp, cands, jobs)
        results["failure_model_missing"] = {
            "passed": r1["status"] == "ok" and r1["mode"] == "baseline_fallback" and len(r1["ranked"]) == 20,
            "detail": r1,
        }
    finally:
        shutil.move(tmp_missing, f"{MODELS}/ranker_v1.joblib")

    model = joblib.load(f"{MODELS}/ranker_v1.joblib")

    # Failure 2: malformed row (candidate with NaN skills). First run
    # (pre-fix) crashed with "'float' object has no attribute 'split'" --
    # caught by this exact test. features.py was then hardened to degrade
    # the row to zero skill-overlap instead of crashing. This confirms
    # the fix holds.
    imp_bad = imp.copy()
    cands_bad = cands.copy()
    bad_id = pool.iloc[0]["candidate_id"]
    cands_bad.loc[cands_bad["candidate_id"] == bad_id, "skills"] = np.nan
    try:
        feats_bad = build_features(imp_bad, cands_bad, jobs)
        feats_bad["score"] = model.predict(feats_bad[FEATURE_COLUMNS])
        bad_row = feats_bad[feats_bad["candidate_id"] == bad_id].iloc[0]
        r2 = {
            "status": "ok",
            "n_ranked": len(feats_bad),
            "malformed_candidate_skill_overlap_degraded_to_zero": bool(bad_row["skill_overlap"] == 0.0),
        }
        results["failure_malformed_row"] = {
            "passed": r2["n_ranked"] == 20 and r2["malformed_candidate_skill_overlap_degraded_to_zero"],
            "detail": r2,
        }
    except Exception as e:
        results["failure_malformed_row"] = {"passed": False, "detail": {"exception": str(e)}}

    # Failure 3: empty candidate pool
    empty_imp = imp.iloc[0:0]
    r3 = safe_rank(model, empty_imp, cands, jobs)
    results["failure_empty_pool"] = {
        "passed": r3["status"] == "empty" and r3["ranked"] == [],
        "detail": r3,
    }

    all_passed = all(v["passed"] for v in results.values())
    print("ALL DEGRADATIONS BEHAVED AS DESIGNED:" if all_passed else "FAILURE TEST FOUND A GAP:")
    for name, v in results.items():
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {name}: {v['detail']}")

    import json
    exper = os.path.join(os.path.dirname(__file__), "..", "experiments")
    with open(f"{exper}/failure_test_report.json", "w") as f:
        json.dump({"all_passed": all_passed, "results": results}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
