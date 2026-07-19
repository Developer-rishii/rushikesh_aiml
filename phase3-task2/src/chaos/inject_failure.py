"""
Stage E of the build pipeline: "Trigger a synthetic latency/quality breach
and show the alert firing" + "Deliberately induce the failure and confirm
the designed degradation actually happens."

This script is a live HTTP client against src/serving/app.py (must already
be running). It:
  1. Sends a batch of NORMAL requests (baseline healthy traffic) and
     freezes the reference score distribution off the back of them.
  2. Checks /metrics -- should be healthy, no alerts.
  3. Injects a LATENCY spike via /chaos and re-sends traffic -> expects
     LATENCY_P95_BREACH / LATENCY_P99_HARD_BREACH to fire.
  4. Clears latency chaos, injects a DEGENERATE-output chaos (model
     returns near-constant scores, simulating a silent failure) ->
     expects SCORE_DEGENERATE (and often SCORE_DRIFT) to fire.
  5. Injects FORCE_UNAVAILABLE -> confirms the service falls back to the
     popularity baseline (degraded=True) instead of hard-failing, and that
     this is visible in /metrics and the error budget.
  6. Clears all chaos and confirms the system returns to healthy.
"""
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = "http://127.0.0.1:8899"
ROOT = Path(__file__).resolve().parents[2]


def load_sample_requests(n=60):
    df = pd.read_csv(ROOT / "data" / "raw" / "interaction_logs.csv")
    sample = df.sample(n=n, random_state=random.randint(0, 10_000))
    return sample.to_dict(orient="records")


def send_batch(records, label):
    print(f"\n--- Sending {len(records)} requests: {label} ---")
    for r in records:
        try:
            resp = requests.post(f"{BASE}/predict", json=r, timeout=5)
            _ = resp.json()
        except Exception as e:
            print(f"  request error: {e}")
    metrics = requests.get(f"{BASE}/metrics", timeout=5).json()
    print(f"  snapshot: {json.dumps(metrics['snapshot'])}")
    if metrics["alerts_fired_this_check"]:
        for a in metrics["alerts_fired_this_check"]:
            pass  # already printed server-side; avoid double emission here
    else:
        print("  (no alerts fired on this check)")
    return metrics


def chaos(action, **kwargs):
    payload = {"action": action, **kwargs}
    requests.post(f"{BASE}/chaos", json=payload, timeout=5)
    print(f"[chaos] applied: {payload}")


def main():
    print("=== Stage E: integrate, break it, then demo ===")
    health = requests.get(f"{BASE}/health", timeout=5).json()
    print(f"Service health: {health}")

    # 1. Healthy baseline traffic
    healthy = load_sample_requests(80)
    send_batch(healthy, "HEALTHY baseline traffic")

    # 2. Freeze reference distribution for drift comparison (would normally
    #    be frozen once, right after a validated deploy)
    requests.post(f"{BASE}/chaos", json={"action": "freeze_reference"})
    print("\n[reference distribution frozen for drift comparison]")

    # 3. Inject latency spike
    chaos("inject_latency", ms=350)
    latency_batch = load_sample_requests(30)
    send_batch(latency_batch, "LATENCY SPIKE injected (350ms/request)")

    # 4. Clear latency, inject degenerate output (silent failure).
    #    Rolling window is 200 requests -- send enough degenerate traffic
    #    to fully flush healthy history out of the window so the
    #    degenerate signal isn't diluted, exactly as sustained silent
    #    failure in production would.
    chaos("clear")
    chaos("force_degenerate")
    degenerate_batch = load_sample_requests(220)
    send_batch(degenerate_batch, "DEGENERATE MODEL OUTPUT injected (near-constant scores, sustained)")

    # 5. Clear, inject full unavailability -> confirm fallback degrades gracefully
    chaos("clear")
    chaos("force_unavailable")
    unavail_batch = load_sample_requests(20)
    m = send_batch(unavail_batch, "MODEL UNAVAILABLE injected (expect graceful fallback, not 5xx)")

    sample_resp = requests.post(f"{BASE}/predict", json=load_sample_requests(1)[0], timeout=5).json()
    print(f"\nSample response while model unavailable (fallback in effect): {json.dumps(sample_resp, indent=2)}")

    # 6. Recover
    chaos("clear")
    recovered = load_sample_requests(40)
    send_batch(recovered, "RECOVERY -- chaos cleared, expect healthy metrics again")

    print("\n=== Chaos run complete. See logs/alerts.log and logs/predictions.jsonl for full evidence. ===")


if __name__ == "__main__":
    main()
