#!/usr/bin/env bash
# Runs the entire Task 2 pipeline end-to-end: data -> train -> evaluate ->
# fairness -> serve -> break it -> error budget. Safe to re-run repeatedly
# (regenerates data, retrains a fresh versioned model, appends to the
# experiment log, and produces a fresh evidence trail under logs/ and docs/).
set -e
cd "$(dirname "$0")"

echo "=== [1/7] Generating realistic interaction logs ==="
python3 src/data_generation/generate_logs.py

echo -e "\n=== [2/7] Train/serve consistency tests (guards against feature skew) ==="
python3 -c "
from tests.test_train_serve_consistency import *
test_single_row_matches_batch_computation()
test_serving_request_shape_produces_same_features_as_training_row()
print('Both train/serve consistency tests PASSED')
"

echo -e "\n=== [3/7] Training model (time-based split, vs popularity baseline) ==="
python3 src/training/train_model.py

echo -e "\n=== [4/7] Standalone evaluation report ==="
python3 src/training/evaluate_model.py

echo -e "\n=== [5/7] Fairness sanity check ==="
python3 src/fairness/fairness_check.py

echo -e "\n=== [6/7] Starting live service, running healthy + chaos traffic ==="
rm -f logs/predictions.jsonl logs/alerts.log logs/service_stdout.log
python3 -W ignore src/serving/app.py > logs/service_stdout.log 2>&1 &
SERVER_PID=$!
sleep 2
curl -s http://127.0.0.1:8899/health
echo ""
python3 src/chaos/inject_failure.py
kill $SERVER_PID 2>/dev/null || true

echo -e "\n=== [7/7] Computing error budget from this run's real logs ==="
python3 src/slo/error_budget.py

echo -e "\n=== Done. See logs/alerts.log, logs/predictions.jsonl, and docs/ for full evidence. ==="
