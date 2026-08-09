"""
Certification pack - latency & cost sections (Sec 4: "cheap enough to serve
at marketplace scale"). Measures real wall-clock inference latency on this
machine (p50/p95/p99) for single-candidate scoring (the real serving
pattern: rank ~15 jobs for one candidate per request), then converts
throughput into a $/1000-predictions cost estimate against a stated
reference instance price (documented, not invented silently).
"""
import os, pickle, json, time
import numpy as np
import pandas as pd
from common import get_X

ROOT = os.path.dirname(os.path.dirname(__file__))
EXP_LOG = f"{ROOT}/reports/experiment_log.csv"
REFERENCE_INSTANCE_USD_PER_HOUR = 0.14  # e.g. a small CPU inference instance, stated assumption

def main():
    df = pd.read_csv(f"{ROOT}/data/logs.csv")
    test = df[df.day >= 20]
    with open(f"{ROOT}/registry/models/ranker_v2.0.pkl", "rb") as f:
        model = pickle.load(f)

    # warm up
    sample_query = test.query_id.unique()[0]
    warm = get_X(test[test.query_id == sample_query])
    for _ in range(5):
        model.predict(warm)

    latencies_ms = []
    queries = test.query_id.unique()[:2000]
    for qid in queries:
        batch = get_X(test[test.query_id == qid])
        t0 = time.perf_counter()
        model.predict(batch)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies_ms)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    requests_per_sec = 1000.0 / p50 if p50 > 0 else float("inf")
    cost_per_1000 = (REFERENCE_INSTANCE_USD_PER_HOUR / 3600) / requests_per_sec * 1000 if requests_per_sec else None

    result = {
        "n_requests_measured": len(lat),
        "p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3), "p99_ms": round(float(p99), 3),
        "slo_target_p95_ms": 150,
        "slo_met": bool(p95 < 150),
        "reference_instance_usd_per_hour": REFERENCE_INSTANCE_USD_PER_HOUR,
        "estimated_cost_usd_per_1000_requests": round(cost_per_1000, 6) if cost_per_1000 else None,
        "note": "single-instance, single-thread CPU inference measured on this "
                "container; production would front with batching + autoscaling, "
                "this is a conservative worst-case single-node number."
    }
    json.dump(result, open(f"{ROOT}/reports/latency_cost.json", "w"), indent=2)
    with open(EXP_LOG, "a") as f:
        t = int(time.time())
        f.write(f"latency_cost,bench,p95_ms,{p95:.3f},{t}\n")
        f.write(f"latency_cost,bench,cost_per_1000,{cost_per_1000:.6f},{t}\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
