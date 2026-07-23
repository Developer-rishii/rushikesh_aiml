"""
Failure injection demonstration script.
Runs a sequence of requests: normal → forced-failure → normal (recovery),
and saves results to results/failure_demo_log.json with a computed PASS/FAIL verdict.
"""
import requests
import time
import json
import os
import sys

SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAYLOAD = {
    "candidate_id": "C75",
    "candidate_exp": 5,
    "candidate_skills": 4,
    "jobs": [
        {"job_id": "J12", "required_exp": 3, "required_skills": 2, "job_popularity": 0.8},
        {"job_id": "J15", "required_exp": 8, "required_skills": 6, "job_popularity": 0.9},
    ]
}

def make_request(inject_failure=False):
    headers = {"Content-Type": "application/json"}
    if inject_failure:
        headers["x-fail-model"] = "true"

    start = time.perf_counter()
    resp = requests.post(f"{SERVICE_URL}/predict", json=PAYLOAD, headers=headers, timeout=5)
    elapsed_ms = (time.perf_counter() - start) * 1000

    data = resp.json()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status_code": resp.status_code,
        "latency_ms": round(elapsed_ms, 2),
        "used_fallback": data.get("used_fallback"),
        "fallback_reason": data.get("fallback_reason"),
        "ranked_jobs": data.get("ranked_jobs"),
        "inject_failure": inject_failure,
    }

def main():
    root = get_project_root()
    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    log = {"phases": [], "verdict": {}}

    # Phase 1: Normal operation (5 requests)
    print("Phase 1: Normal operation (no failure injection)")
    phase1 = []
    for i in range(5):
        r = make_request(inject_failure=False)
        phase1.append(r)
        print(f"  [{i+1}] status={r['status_code']} latency={r['latency_ms']:.1f}ms fallback={r['used_fallback']}")
    log["phases"].append({"name": "normal_before", "requests": phase1})

    # Phase 2: Failure injection (5 requests)
    print("\nPhase 2: Failure injection (x-fail-model: true)")
    phase2 = []
    for i in range(5):
        r = make_request(inject_failure=True)
        phase2.append(r)
        print(f"  [{i+1}] status={r['status_code']} latency={r['latency_ms']:.1f}ms fallback={r['used_fallback']} reason={r['fallback_reason']}")
    log["phases"].append({"name": "failure_injected", "requests": phase2})

    # Phase 3: Recovery (5 requests, no injection)
    print("\nPhase 3: Recovery (failure injection removed)")
    phase3 = []
    for i in range(5):
        r = make_request(inject_failure=False)
        phase3.append(r)
        print(f"  [{i+1}] status={r['status_code']} latency={r['latency_ms']:.1f}ms fallback={r['used_fallback']}")
    log["phases"].append({"name": "normal_after_recovery", "requests": phase3})

    # Compute verdict
    all_200 = all(r['status_code'] == 200 for phase in log['phases'] for r in phase['requests'])
    normal_no_fallback = all(r['used_fallback'] == False for r in phase1 + phase3)
    injected_all_fallback = all(r['used_fallback'] == True for r in phase2)
    injected_all_forced = all(r['fallback_reason'] == 'forced' for r in phase2)
    recovery_confirmed = all(r['used_fallback'] == False for r in phase3)

    verdict = {
        "all_requests_returned_200": all_200,
        "normal_requests_used_model": normal_no_fallback,
        "injected_requests_used_fallback": injected_all_fallback,
        "injected_reason_is_forced": injected_all_forced,
        "recovery_confirmed": recovery_confirmed,
        "PASS": all_200 and normal_no_fallback and injected_all_fallback and injected_all_forced and recovery_confirmed
    }
    log["verdict"] = verdict

    print(f"\n--- VERDICT ---")
    for k, v in verdict.items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}: {v}")

    out_path = os.path.join(results_dir, 'failure_demo_log.json')
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
