"""
load_test.py
-------------
Drives the /rank endpoint at increasing concurrency levels and records,
for each level: achieved QPS, p50/p95/p99 client-side latency, error rate,
and fallback rate (how often the service had to degrade instead of using
the real model). This is stage E.2 of the build pipeline: "Run the load
test live; show latency curves and the failure point."

Usage:
  python3 src/load_test.py --url http://127.0.0.1:8877/rank --out results/load_test_results.csv
"""
import argparse
import concurrent.futures
import json
import statistics
import time

import requests

CANDIDATES = [
    {"candidate_id": f"c{i}", "exp_years": 5 + (i % 10), "skill_match": 0.4 + 0.01 * (i % 50),
     "education_score": 0.5, "location_match": i % 2, "past_response_rate": 0.3}
    for i in range(25)
]
JOB = {"job_seniority": 2, "job_urgency_score": 0.6, "job_num_applicants_so_far": 40}


def one_request(url, timeout_s=3.0):
    t0 = time.time()
    try:
        r = requests.post(url, json={"candidates": CANDIDATES, "job": JOB}, timeout=timeout_s)
        latency_ms = (time.time() - t0) * 1000
        ok = r.status_code == 200
        body = r.json() if ok else {}
        return {
            "latency_ms": latency_ms,
            "ok": ok,
            "used_fallback": body.get("used_fallback", False),
        }
    except requests.exceptions.RequestException:
        latency_ms = (time.time() - t0) * 1000
        return {"latency_ms": latency_ms, "ok": False, "used_fallback": None}


def run_level(url, concurrency, duration_s):
    """Fire requests with `concurrency` workers continuously for duration_s seconds."""
    results = []
    stop_at = time.time() + duration_s

    def worker_loop():
        local = []
        while time.time() < stop_at:
            local.append(one_request(url))
        return local

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker_loop) for _ in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
    return results


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[idx]


def summarize(level, results, duration_s):
    n = len(results)
    lat = [r["latency_ms"] for r in results]
    errors = sum(1 for r in results if not r["ok"])
    fallbacks = sum(1 for r in results if r.get("used_fallback"))
    return {
        "concurrency": level,
        "n_requests": n,
        "achieved_qps": round(n / duration_s, 1),
        "p50_ms": round(pct(lat, 50), 1),
        "p95_ms": round(pct(lat, 95), 1),
        "p99_ms": round(pct(lat, 99), 1),
        "max_ms": round(max(lat), 1) if lat else None,
        "error_rate": round(errors / n, 4) if n else None,
        "fallback_rate": round(fallbacks / n, 4) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8877/rank")
    ap.add_argument("--out", default="results/load_test_results.csv")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds per concurrency level")
    ap.add_argument("--levels", default="1,2,4,8,16,24,32,48,64,96,128")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    # Warm the model once so the load curve reflects steady-state serving,
    # not the one-time cold start (that is measured separately).
    one_request(args.url)
    time.sleep(0.2)

    rows = []
    for level in levels:
        res = run_level(args.url, level, args.duration)
        row = summarize(level, res, args.duration)
        rows.append(row)
        print(row)

    with open(args.out, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
