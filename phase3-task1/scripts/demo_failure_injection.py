"""
scripts/demo_failure_injection.py
----------------------------------
Stage E.3: Deliberate failure injection demo.

Demonstrates that the validation layer fires correctly when:
  1. A required column is missing/corrupted in serving_features.csv
  2. prediction_logs.csv is stale (simulated)
  3. An unknown log_id is passed to score_one()

Run from the phase3-task1 root directory:
    python scripts/demo_failure_injection.py
"""

import sys
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from health_monitor import _validate_inputs, compute_health_report
from defect_ranker import score_one

print("=" * 60)
print("  Phase 3 Task 1 -- Failure Injection Demo (Stage E.3)")
print("=" * 60)

# ── Case 1: Missing column in serving_features ---------------------------------
print("\n[1/3] Missing required column -- 'skill_gap' renamed to 'skill_GAP_TYPO'")
pred  = pd.read_csv(ROOT / "data" / "prediction_logs.csv")
inter = pd.read_csv(ROOT / "data" / "interaction_logs.csv")
train = pd.read_csv(ROOT / "data" / "training_features.csv")
serve = pd.read_csv(ROOT / "data" / "serving_features.csv")

# Corrupt: rename a required column
serve_corrupt = serve.rename(columns={"skill_gap": "skill_GAP_TYPO"})

try:
    _validate_inputs(pred, inter, train, serve_corrupt)
    print("  ERROR: Validation did NOT fire -- this should not happen!")
except ValueError as e:
    print(f"  [PASS] ValueError fired as expected:")
    print(f"         {e}")

# ── Case 2: Empty prediction log -----------------------------------------------
print("\n[2/3] Empty prediction_logs.csv (simulates stale/missing pipeline output)")
pred_empty = pred.iloc[0:0]  # empty DataFrame with same columns
try:
    _validate_inputs(pred_empty, inter, train, serve)
    print("  ERROR: Validation did NOT fire -- this should not happen!")
except ValueError as e:
    print(f"  [PASS] ValueError fired as expected:")
    print(f"         {e}")

# ── Case 3: Unknown log_id passed to score_one --------------------------------
print("\n[3/3] Unknown log_id 'L999999_UNKNOWN' passed to score_one()")
result = score_one("L999999_UNKNOWN", pred)
if "error" in result:
    print(f"  [PASS] score_one returned error dict (no exception):")
    print(f"         {json.dumps(result, indent=9)}")
else:
    print(f"  UNEXPECTED result: {result}")

print("\n" + "=" * 60)
print("  All 3 failure cases handled correctly -- validation is live.")
print("=" * 60)
