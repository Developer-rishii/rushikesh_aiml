#!/usr/bin/env bash
# run_all.sh -- the full Task 11 pipeline, end to end, in one command.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

echo "== Stage 1/5: generating logged impressions (simulated marketplace logs) =="
python3 "$ROOT/data/generate_logs.py"

echo -e "\n== Stage 2/5: estimating position-bias propensities (intervention harvesting) =="
python3 "$ROOT/src/position_bias.py"

echo -e "\n== Stage 3/5: training LTR models + offline evaluation vs heuristic =="
python3 "$ROOT/src/evaluate.py"

echo -e "\n== Stage 4/5: fairness parity + drift monitoring =="
python3 "$ROOT/src/fairness_drift_runner.py"

echo -e "\n== Stage 5/5: failure-mode + regression tests (deliberately induced failures) =="
python3 "$ROOT/tests/test_failure_and_bias.py"

echo -e "\nDone. See reports/metrics.json, reports/fairness_drift.json, reports/*.md"
