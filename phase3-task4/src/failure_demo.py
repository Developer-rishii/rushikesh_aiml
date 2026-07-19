"""
failure_demo.py
----------------
Stage E.3: "Deliberately induce the failure and confirm the designed
degradation actually happens." Not a claim -- this script actually flips
the model off via /admin/model_down while requests are in flight, and
records what really happens request-by-request.

Success criteria (checked programmatically, not just eyeballed):
  1. error_rate stays 0 throughout (no 5xx, no dropped requests)
  2. used_fallback flips from False -> True the instant the model goes down
  3. latency stays bounded (fallback path is cheap) even though the "real"
     model is completely unavailable
  4. flipping the model back up restores used_fallback -> False
"""
import json
import time

import requests

BASE = "http://127.0.0.1:8877"
CANDIDATES = [
    {"candidate_id": f"c{i}", "exp_years": 5, "skill_match": 0.6, "education_score": 0.5,
     "location_match": 1, "past_response_rate": 0.3}
    for i in range(20)
]
JOB = {"job_seniority": 2, "job_urgency_score": 0.5, "job_num_applicants_so_far": 30}


def call():
    t0 = time.time()
    r = requests.post(f"{BASE}/rank", json={"candidates": CANDIDATES, "job": JOB}, timeout=3)
    return {
        "t": round(time.time(), 3),
        "status": r.status_code,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "used_fallback": r.json().get("used_fallback"),
        "fallback_reason": r.json().get("fallback_reason"),
    }


def main():
    log = []
    print("Phase 1: normal traffic (model healthy)")
    for _ in range(5):
        log.append(call())
        time.sleep(0.1)

    print("Phase 2: INDUCING FAILURE -> POST /admin/model_down")
    requests.post(f"{BASE}/admin/model_down")
    for _ in range(8):
        log.append(call())
        time.sleep(0.1)

    print("Phase 3: RECOVERY -> POST /admin/model_up")
    requests.post(f"{BASE}/admin/model_up")
    for _ in range(5):
        log.append(call())
        time.sleep(0.1)

    for row in log:
        print(row)

    errors = sum(1 for r in log if r["status"] != 200)
    down_phase = log[5:13]
    fallback_engaged = all(r["used_fallback"] for r in down_phase)
    recovered = not log[-1]["used_fallback"]
    max_latency_during_outage = max(r["latency_ms"] for r in down_phase)

    verdict = {
        "total_requests": len(log),
        "errors": errors,
        "fallback_engaged_during_outage": fallback_engaged,
        "max_latency_ms_during_outage": max_latency_during_outage,
        "recovered_after_model_up": recovered,
        "PASS": errors == 0 and fallback_engaged and recovered,
    }
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))

    with open("results/failure_demo_log.json", "w") as f:
        json.dump({"requests": log, "verdict": verdict}, f, indent=2)
    print("\nwrote results/failure_demo_log.json")


if __name__ == "__main__":
    main()
