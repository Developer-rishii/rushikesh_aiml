"""
failure_injection.py
Stage E.3: "Deliberately induce the failure and confirm the designed
degradation actually happens." Covers threat T6 (model unavailability).

Design: if the ranking model raises/times out, the serving layer MUST fall
back to a deterministic recency-based ranking rather than crashing or
returning an unranked/empty result -- verified here, not just claimed.
"""
import json, os, sys, random

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "ranking_defense"))
from ranker import robust_features


class FlakyModel:
    """Wraps the real model but simulates an outage on demand."""
    def __init__(self, real_model, simulate_down=False):
        self.real_model = real_model
        self.simulate_down = simulate_down

    def predict(self, X):
        if self.simulate_down:
            raise RuntimeError("SIMULATED MODEL OUTAGE")
        return self.real_model.predict(X)


def recency_fallback_rank(candidates_subset):
    """Deterministic, model-free fallback: sort by years_exp desc (proxy for
    a 'recency/seniority' fallback signal) then candidate_id for determinism."""
    return sorted(candidates_subset, key=lambda c: (-c["years_exp"], c["candidate_id"]))


def safe_rank(candidates_subset, job, model, stuffing_scores):
    """Serving-layer wrapper: try the ML ranker; on ANY exception, fall back
    to the deterministic recency ranker and flag degraded mode. Never lets
    an exception propagate to the caller / never returns an empty result."""
    try:
        scored = []
        for c in candidates_subset:
            s = stuffing_scores.get(c["candidate_id"], 0.0)
            pred = model.predict([robust_features(c, job, s)])[0]
            scored.append((pred, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored], "ml_ranker"
    except Exception as e:
        fallback = recency_fallback_rank(candidates_subset)
        return fallback, f"degraded_recency_fallback (reason: {e})"


def main():
    import joblib
    with open(os.path.join(ROOT, "data", "candidates.json")) as f:
        candidates = json.load(f)[:20]
    with open(os.path.join(ROOT, "data", "jobs.json")) as f:
        jobs = json.load(f)
    job = jobs[0]
    real_model = joblib.load(os.path.join(ROOT, "ranking_defense", "ranker.joblib"))
    stuffing_scores = {c["candidate_id"]: 0.0 for c in candidates}

    print("=== Normal operation ===")
    healthy_model = FlakyModel(real_model, simulate_down=False)
    ranked, mode = safe_rank(candidates, job, healthy_model, stuffing_scores)
    print(f"  mode={mode}, top_candidate={ranked[0]['candidate_id']}, "
          f"n_results={len(ranked)}")
    ok_normal = mode == "ml_ranker" and len(ranked) == len(candidates)

    print("\n=== Induced failure: model outage ===")
    down_model = FlakyModel(real_model, simulate_down=True)
    ranked_fallback, mode_fallback = safe_rank(candidates, job, down_model, stuffing_scores)
    print(f"  mode={mode_fallback}, top_candidate={ranked_fallback[0]['candidate_id']}, "
          f"n_results={len(ranked_fallback)}")
    ok_fallback = mode_fallback.startswith("degraded_recency_fallback") and \
        len(ranked_fallback) == len(candidates)

    result = {
        "normal_operation_pass": ok_normal,
        "failure_degradation_pass": ok_fallback,
        "overall_pass": ok_normal and ok_fallback,
    }
    print(f"\nRESULT: {'PASS' if result['overall_pass'] else 'FAIL'} -- "
          f"system serves a full, non-empty, deterministic result set even "
          f"when the ML model is completely unavailable")

    with open(os.path.join(ROOT, "integration", "failure_injection_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result["overall_pass"]


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
