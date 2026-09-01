"""
load_test.py
------------
Measures inference latency under realistic repeated load (not a single
happy-path call). Sends 300 sequential real-shaped requests to the live
/predict endpoint and reports p50/p95/p99 latency.

Run (with the server already up on localhost:5000):
    python tests/load_test.py
"""

import json
import statistics
import time
from pathlib import Path

import requests
from sklearn.datasets import load_breast_cancer

BASE_URL = "http://127.0.0.1:5000"
N_REQUESTS = 300
ROOT = Path(__file__).resolve().parent.parent


def main():
    data = load_breast_cancer()
    X = data.data.tolist()
    n = len(X)

    server_latencies = []
    wall_latencies = []

    for i in range(N_REQUESTS):
        feats = X[i % n]
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/predict", json={"features": feats}, timeout=5)
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_latencies.append(wall_ms)
        if r.status_code == 200:
            server_latencies.append(r.json()["latency_ms"])

    def pct(vals, p):
        vals = sorted(vals)
        idx = int(len(vals) * p / 100)
        idx = min(idx, len(vals) - 1)
        return vals[idx]

    report = {
        "n_requests": N_REQUESTS,
        "successful": len(server_latencies),
        "server_side_inference_ms": {
            "mean": round(statistics.mean(server_latencies), 3),
            "p50": round(pct(server_latencies, 50), 3),
            "p95": round(pct(server_latencies, 95), 3),
            "p99": round(pct(server_latencies, 99), 3),
            "max": round(max(server_latencies), 3),
        },
        "wall_clock_roundtrip_ms": {
            "mean": round(statistics.mean(wall_latencies), 3),
            "p50": round(pct(wall_latencies, 50), 3),
            "p95": round(pct(wall_latencies, 95), 3),
            "p99": round(pct(wall_latencies, 99), 3),
            "max": round(max(wall_latencies), 3),
        },
        "acceptable_for_realtime_use": pct(wall_latencies, 95) < 100,
    }

    with open(ROOT / "logs" / "latency_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
