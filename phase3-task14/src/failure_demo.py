"""
Stage E - Integrate, break it, then demo.

Uses Flask's test client (no network needed) to exercise the real
/explain API exactly as a caller would, then deliberately flips the
model into an unavailable state and confirms the designed degradation
(explicit DEFERRED_TO_HUMAN_REVIEW - never a silent guess) happens.

Run: python3 failure_demo.py
Output is also written to reports/demo_transcript.txt for evidence.
"""
import json
import os
import pandas as pd
from paths import EXPERIMENTS_DIR, REPORTS_DIR
import api

lines = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)


client = api.app.test_client()

log("=" * 70)
log("STEP 1: /health before anything happens")
log(json.dumps(client.get("/health").get_json(), indent=2))

log("\n" + "=" * 70)
log("STEP 2: worked example - a REAL held-out candidate, per-decision explanation")
test_df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "test_predictions_mitigated.csv"))
sample = test_df.iloc[0]
payload = {f: (int(sample[f]) if f in ("applications_count", "college_tier", "pincode_tier") else float(sample[f]))
           for f in api.MODEL_FEATURES}
log("Candidate features sent to API:", json.dumps(payload, indent=2))
resp = client.post("/explain", json=payload)
result = resp.get_json()
log(f"\nStatus: {resp.status_code}")
log("Model decision:", result["decision"], "| probability:", result["probability"])
log("Ground truth label was:", "shortlisted" if sample["shortlisted"] == 1 else "not shortlisted")
log("\nPlain-English explanation:")
log(result["explanation"])

log("\n" + "=" * 70)
log("STEP 3: DELIBERATELY INDUCE FAILURE (model unavailable)")
inj = client.post("/admin/inject_failure", json={"unavailable": True})
log(json.dumps(inj.get_json(), indent=2))

log("\nSTEP 4: same request again, model now unavailable")
resp2 = client.post("/explain", json=payload)
log(f"Status: {resp2.status_code}  (expect 503, NOT a silent 200 with a fabricated score)")
log(json.dumps(resp2.get_json(), indent=2))

assert resp2.status_code == 503
assert resp2.get_json()["decision"] == "DEFERRED_TO_HUMAN_REVIEW"
log("\nPASS: system degrades explicitly to human review, never fabricates a decision.")

log("\n" + "=" * 70)
log("STEP 5: restore service, confirm recovery")
client.post("/admin/inject_failure", json={"unavailable": False})
resp3 = client.post("/explain", json=payload)
log(f"Status: {resp3.status_code}  decision: {resp3.get_json()['decision']}")
assert resp3.status_code == 200
log("PASS: service recovered.")

with open(os.path.join(REPORTS_DIR, "demo_transcript.txt"), "w") as f:
    f.write("\n".join(lines))
log("\nTranscript saved to reports/demo_transcript.txt")
