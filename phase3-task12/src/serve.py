"""
Stage D: "Serving path within the latency SLO."
Also implements Stage B/C/D step 4's "what happens when the model is unavailable" --
a real serving path must degrade safely, not throw a 500 to the frontend.

SLO: p95 latency < 150ms per candidate recommendation request (single-model,
in-process; a real deployment would put this behind a feature store + vector
index, noted in "Go deeper").
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from recommender import recommend_for_candidate
from baseline import popularity_baseline_topk

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "model_registry")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_log.json")
SLO_MS = 150


class RecommenderService:
    def __init__(self):
        with open(LOG_PATH) as f:
            self.entry = json.load(f)[-1]
        self.model = joblib.load(self.entry["model_path"])
        self.candidates = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
        self.jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
        self._fallback_cache = popularity_baseline_topk(self.jobs, k=10)
        self.model_available = True  # toggled by failure_test.py to simulate an outage

    def get_recommendations(self, candidate_id, k=10):
        start = time.perf_counter()
        try:
            if not self.model_available:
                raise RuntimeError("simulated model unavailability")
            recs = recommend_for_candidate(self.model, candidate_id, self.candidates, self.jobs, k=k)
            mode = "personalized"
        except Exception as e:
            # designed degradation: serve cached popularity fallback instead of erroring out
            recs = [{"job_id": jid, "relevance_score": None, "two_sided_score": None,
                      "reason": "Popular job shown while personalization is temporarily unavailable."}
                     for jid in self._fallback_cache[:k]]
            mode = f"fallback ({type(e).__name__})"
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "candidate_id": candidate_id,
            "mode": mode,
            "latency_ms": round(latency_ms, 2),
            "within_slo": latency_ms < SLO_MS,
            "recommendations": recs,
        }


def load_latency_benchmark(service, n=100):
    sample_ids = service.candidates["candidate_id"].sample(n, random_state=1).tolist()
    latencies = []
    for cid in sample_ids:
        r = service.get_recommendations(cid)
        latencies.append(r["latency_ms"])
    latencies = np.array(latencies)
    return {
        "n_requests": n,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(latencies.max()),
        "slo_ms": SLO_MS,
        "within_slo_p95": float(np.percentile(latencies, 95)) < SLO_MS,
    }


def main():
    service = RecommenderService()
    bench = load_latency_benchmark(service, n=100)
    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                             f"serving_latency_{service.entry['version']}.json")
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
