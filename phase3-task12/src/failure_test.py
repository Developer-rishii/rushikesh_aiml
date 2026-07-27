"""
Stage E step 3: "Deliberately induce the failure and confirm the designed
degradation actually happens." Not a claim -- this actually flips the model
off, calls the service, and asserts the fallback path fired and stayed within
a relaxed SLO (fallback path is cheaper than the model path, so it should
still pass easily).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from serve import RecommenderService, SLO_MS


def run():
    service = RecommenderService()
    sample_id = service.candidates["candidate_id"].iloc[0]

    normal = service.get_recommendations(sample_id)
    assert normal["mode"] == "personalized", "expected personalized mode before failure injection"
    assert len(normal["recommendations"]) == 10

    # inject failure
    service.model_available = False
    degraded = service.get_recommendations(sample_id)
    assert degraded["mode"].startswith("fallback"), "fallback did not trigger on simulated outage"
    assert len(degraded["recommendations"]) == 10, "fallback must still return a full slate"
    assert degraded["latency_ms"] < SLO_MS, "fallback path must also respect the SLO"

    # recovery
    service.model_available = True
    recovered = service.get_recommendations(sample_id)
    assert recovered["mode"] == "personalized", "service did not recover after failure cleared"

    report = {
        "test": "model_unavailability_failure_injection",
        "result": "PASS",
        "before_failure_mode": normal["mode"],
        "during_failure_mode": degraded["mode"],
        "during_failure_sample_recs": [r["job_id"] for r in degraded["recommendations"]],
        "after_recovery_mode": recovered["mode"],
        "degraded_latency_ms": degraded["latency_ms"],
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "failure_injection_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
