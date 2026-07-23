"""
Custom load test script that runs against the FastAPI service at multiple
concurrency levels, records per-request data, and generates:
  - results/load_test_results.csv (raw per-request data)
  - results/breaking_point_report.md (p50/p95/p99 at each level)

Distinguishes real-model vs fallback-path latency by checking the
used_fallback field in each response.
"""
import requests
import time
import csv
import os
import sys
import json
import concurrent.futures
import statistics

SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAYLOAD_NORMAL = {
    "candidate_id": "C75",
    "candidate_exp": 5,
    "candidate_skills": 4,
    "jobs": [
        {"job_id": "J12", "required_exp": 3, "required_skills": 2, "job_popularity": 0.8},
        {"job_id": "J15", "required_exp": 8, "required_skills": 6, "job_popularity": 0.2},
        {"job_id": "J20", "required_exp": 2, "required_skills": 1, "job_popularity": 0.5},
        {"job_id": "J30", "required_exp": 5, "required_skills": 4, "job_popularity": 0.9},
        {"job_id": "J40", "required_exp": 1, "required_skills": 3, "job_popularity": 0.3},
    ]
}

def make_request(inject_failure=False):
    """Make a single request and return timing + response info."""
    headers = {"Content-Type": "application/json"}
    if inject_failure:
        headers["x-fail-model"] = "true"

    start = time.perf_counter()
    try:
        resp = requests.post(f"{SERVICE_URL}/predict", json=PAYLOAD_NORMAL, headers=headers, timeout=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        return {
            "status_code": resp.status_code,
            "latency_ms": round(elapsed_ms, 2),
            "used_fallback": data.get("used_fallback", None),
            "fallback_reason": data.get("fallback_reason", None),
            "error": None,
            "inject_failure": inject_failure,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "status_code": 0,
            "latency_ms": round(elapsed_ms, 2),
            "used_fallback": None,
            "fallback_reason": None,
            "error": str(e),
            "inject_failure": inject_failure,
        }

def run_level(concurrency, num_requests_per_worker=20):
    """Run a load test at a given concurrency level."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(concurrency * num_requests_per_worker):
            inject = (i % 5 == 0)  # 20% failure-injected
            futures.append(pool.submit(make_request, inject))
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return results

def percentile(data, pct):
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

def main():
    root = get_project_root()
    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Check service is up
    try:
        health = requests.get(f"{SERVICE_URL}/health", timeout=3)
        print(f"Health check: {health.status_code} - {health.text}")
        if health.status_code != 200:
            print("WARNING: Service health check returned non-200. Model may not be loaded.")
    except Exception as e:
        print(f"ERROR: Cannot reach service at {SERVICE_URL}: {e}")
        sys.exit(1)

    concurrency_levels = [1, 5, 10, 25, 50]
    all_rows = []
    report_lines = []
    report_lines.append("# Load Test Breaking Point Report")
    report_lines.append("")
    report_lines.append(f"Generated programmatically by `scripts/run_load_test.py`.")
    report_lines.append(f"Target: {SERVICE_URL}")
    report_lines.append("")
    report_lines.append("| Concurrency | Total Reqs | Errors | p50 (ms) | p95 (ms) | p99 (ms) | Model p50 (ms) | Fallback p50 (ms) | Fallback Rate |")
    report_lines.append("|-------------|-----------|--------|----------|----------|----------|----------------|-------------------|---------------|")

    for level in concurrency_levels:
        print(f"\n--- Concurrency: {level} ---")
        results = run_level(level, num_requests_per_worker=20)

        all_latencies = [r['latency_ms'] for r in results]
        model_latencies = [r['latency_ms'] for r in results if r['used_fallback'] == False and r['error'] is None]
        fallback_latencies = [r['latency_ms'] for r in results if r['used_fallback'] == True and r['error'] is None]
        errors = [r for r in results if r['error'] is not None or r['status_code'] != 200]
        fallback_rate = len(fallback_latencies) / len(results) * 100 if results else 0

        p50 = percentile(all_latencies, 50)
        p95 = percentile(all_latencies, 95)
        p99 = percentile(all_latencies, 99)
        model_p50 = percentile(model_latencies, 50) if model_latencies else 0
        fallback_p50 = percentile(fallback_latencies, 50) if fallback_latencies else 0

        report_lines.append(
            f"| {level} | {len(results)} | {len(errors)} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {model_p50:.1f} | {fallback_p50:.1f} | {fallback_rate:.1f}% |"
        )

        for r in results:
            r['concurrency'] = level
            all_rows.append(r)

        print(f"  Requests: {len(results)}, Errors: {len(errors)}, p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms")
        print(f"  Model path p50: {model_p50:.1f}ms, Fallback path p50: {fallback_p50:.1f}ms")

    # Determine knee point
    report_lines.append("")
    report_lines.append("## Knee Point Analysis")
    report_lines.append("")
    report_lines.append("The knee point is the concurrency level where p99 latency first exceeds the 50ms SLO.")
    report_lines.append("")

    # Save CSV
    csv_path = os.path.join(results_dir, 'load_test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['concurrency', 'status_code', 'latency_ms', 'used_fallback', 'fallback_reason', 'error', 'inject_failure'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved raw data to {csv_path}")

    # Save report
    report_path = os.path.join(results_dir, 'breaking_point_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
