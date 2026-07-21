"""
2-minute live demo script (Stage E, deliverable 4).
Run with:  python3 -m scripts.demo   (after scripts.run_all has been run once)

Talk track (say this out loud while it prints):
  1. "Here's the BEFORE server: naive per-candidate feature fetch, full model."
  2. "Here's the AFTER server: batched fetch, distilled model, LRU cache."
  3. "Same held-out job, same candidates, same quality bar -- watch the p95."
  4. "Now I kill the model service." -> show the fallback still answers.
"""
import json
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INTERACTIONS_CSV, LATENCY_P95_SLO_MS
from src.utils import ModelRegistry
from src.data_pipeline import FeatureStore
from src.serving import BaselineServer, OptimizedServer, LRUScoreCache, popularity_fallback_scorer
from src.evaluate import evaluate_ranking


def main():
    test_path = INTERACTIONS_CSV.replace(".csv", "_test.csv")
    if not os.path.exists(test_path):
        print("Run `python3 -m scripts.run_all` first to generate data + train models.")
        return
    test_df = pd.read_csv(test_path)
    reg = ModelRegistry()
    baseline_model, base_entry = reg.load_latest("ranker_baseline")
    opt_model, opt_entry = reg.load_latest("ranker_optimized")

    job_id = test_df.job_id.iloc[0]
    rows = test_df[test_df.job_id == job_id].to_dict(orient="records")
    print(f"Demo job: {job_id}  ({len(rows)} candidates)\n")

    # --- BEFORE ---
    fs = FeatureStore(test_df)
    before_server = BaselineServer(baseline_model, fs)
    t0 = time.perf_counter()
    before_scores = before_server.rank(job_id, rows)
    before_ms = (time.perf_counter() - t0) * 1000
    print(f"BEFORE  : {before_ms:6.2f} ms   model={base_entry['file']} "
          f"({base_entry['size_bytes']/1e6:.1f} MB)")

    # --- AFTER ---
    fs2 = FeatureStore(test_df)
    cache = LRUScoreCache()
    after_server = OptimizedServer(opt_model, fs2, cache=cache, fallback_scorer=popularity_fallback_scorer)
    t0 = time.perf_counter()
    after_scores, mode = after_server.rank(job_id, rows, cache_key_fn=lambda r: r["candidate_id"])
    after_ms = (time.perf_counter() - t0) * 1000
    print(f"AFTER   : {after_ms:6.2f} ms   model={opt_entry['file']} "
          f"({opt_entry['size_bytes']/1e3:.1f} KB)   path={mode}")
    print(f"SPEEDUP : {before_ms/after_ms:.1f}x   SLO={LATENCY_P95_SLO_MS} ms\n")

    # --- Quality check on same job, held constant ---
    import numpy as np
    df_before = pd.DataFrame(rows); df_before["score"] = before_scores
    df_after = pd.DataFrame(rows); df_after["score"] = after_scores
    m_before = evaluate_ranking(df_before, "score", "relevance", "job_id", k=10)
    m_after = evaluate_ranking(df_after, "score", "relevance", "job_id", k=10)
    print("Quality on this job -- BEFORE:", m_before)
    print("Quality on this job -- AFTER :", m_after, "\n")

    # --- Failure injection ---
    print("Killing the model service (force_unavailable=True)...")
    failing = OptimizedServer(opt_model, fs2, cache=LRUScoreCache(),
                               fallback_scorer=popularity_fallback_scorer,
                               force_unavailable=True)
    fail_scores, fail_mode = failing.rank(job_id, rows, cache_key_fn=lambda r: r["candidate_id"])
    print(f"Request still served via '{fail_mode}' path, {len(fail_scores)} scores returned, "
          f"no error raised to the caller.")


if __name__ == "__main__":
    main()
