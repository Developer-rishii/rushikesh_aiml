"""
test_api.py
-----------
Live, end-to-end tests against the running Flask service (not mocks).
Covers: health check, real-shaped valid predictions (using actual dataset
rows), batch predictions, and garbage/edge-case input handling.

Run (with the server already up on localhost:5000):
    python tests/test_api.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

BASE_URL = "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = []


def check(name, condition, extra=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status, extra))
    print(f"[{status}] {name} {extra}")
    return condition


def get_real_samples(n=5):
    data = load_breast_cancer()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test[:n].tolist(), y_test[:n].tolist()


def main():
    all_pass = True

    # 1. Health check
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    all_pass &= check("health_check_status_200", r.status_code == 200, f"-> {r.status_code}")
    all_pass &= check("health_check_model_loaded", r.json().get("model_loaded") is True)

    # 2. Real-shaped valid prediction requests (actual dataset rows, unseen split)
    X_real, y_real = get_real_samples(10)
    correct = 0
    latencies = []
    for i, (feats, true_label) in enumerate(zip(X_real, y_real)):
        r = requests.post(f"{BASE_URL}/predict", json={"features": feats}, timeout=5)
        ok = r.status_code == 200
        all_pass &= check(f"real_sample_{i}_status_200", ok, f"-> {r.status_code}")
        if ok:
            body = r.json()
            latencies.append(body["latency_ms"])
            if body["prediction"] == true_label:
                correct += 1
    all_pass &= check(
        "real_sample_predictions_reasonable_accuracy",
        correct / len(X_real) >= 0.7,
        f"-> {correct}/{len(X_real)} correct",
    )
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        all_pass &= check(
            "latency_under_50ms_avg", avg_latency < 50, f"-> avg {avg_latency:.3f} ms"
        )

    # 3. Realistic-scale batch prediction (500 real rows in one call)
    data = load_breast_cancer()
    X_all = data.data.tolist()
    batch_payload = {"instances": [{"features": f} for f in X_all[:500]]}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload, timeout=30)
    batch_elapsed = (time.perf_counter() - t0) * 1000
    all_pass &= check(
        "batch_500_rows_status_ok", r.status_code in (200, 207), f"-> {r.status_code}"
    )
    all_pass &= check(
        "batch_500_rows_all_returned",
        r.json().get("count") == 500,
        f"-> {r.json().get('count')}",
    )
    all_pass &= check("batch_500_rows_latency", batch_elapsed < 5000, f"-> {batch_elapsed:.1f} ms total")

    # 4. Edge case: wrong number of features
    r = requests.post(f"{BASE_URL}/predict", json={"features": [1.0, 2.0]}, timeout=5)
    all_pass &= check("edge_wrong_length_returns_422", r.status_code == 422, f"-> {r.status_code}")

    # 5. Edge case: missing key entirely
    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=5)
    all_pass &= check("edge_missing_key_returns_422", r.status_code == 422, f"-> {r.status_code}")

    # 6. Edge case: non-numeric / garbage values
    bad_feats = ["a"] * 30
    r = requests.post(f"{BASE_URL}/predict", json={"features": bad_feats}, timeout=5)
    all_pass &= check("edge_non_numeric_returns_422", r.status_code == 422, f"-> {r.status_code}")

    # 7. Edge case: NaN injected (sent as raw JSON text since NaN is not
    # standard JSON and Python's json module needs allow_nan to emit it)
    nan_body = json.dumps({"features": [float("nan")] * 30})
    r = requests.post(
        f"{BASE_URL}/predict", data=nan_body, headers={"Content-Type": "application/json"}, timeout=5
    )
    all_pass &= check("edge_nan_returns_400_or_422", r.status_code in (400, 422), f"-> {r.status_code}")

    # 8. Edge case: malformed JSON body
    r = requests.post(
        f"{BASE_URL}/predict", data="not-json", headers={"Content-Type": "application/json"}, timeout=5
    )
    all_pass &= check("edge_malformed_json_returns_400", r.status_code == 400, f"-> {r.status_code}")

    # 9. Edge case: unknown route -> 404 handler
    r = requests.get(f"{BASE_URL}/nonexistent", timeout=5)
    all_pass &= check("edge_unknown_route_returns_404", r.status_code == 404, f"-> {r.status_code}")

    # 10. Edge case: wrong HTTP method -> 405 handler
    r = requests.get(f"{BASE_URL}/predict", timeout=5)
    all_pass &= check("edge_wrong_method_returns_405", r.status_code == 405, f"-> {r.status_code}")

    with open(ROOT / "logs" / "test_results.json", "w") as f:
        json.dump(
            [{"test": n, "status": s, "detail": e} for n, s, e in RESULTS],
            f,
            indent=2,
        )

    print("\n=== SUMMARY ===")
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    print(f"{passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
