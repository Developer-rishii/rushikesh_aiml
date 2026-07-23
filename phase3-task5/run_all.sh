#!/bin/bash
# run_all.sh — Reproduces everything from scratch.
# set -e so it stops on first failure.
set -e

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

echo "========================================"
echo "Phase 3 Task 5: Full Reproduction Run"
echo "========================================"

# Step 1: Generate / validate data
echo ""
echo "[1/7] Generating synthetic data..."
python data/generate_data.py

# Step 2: Train model
echo ""
echo "[2/7] Training LightGBM Ranker..."
python training/train.py

# Step 3: Evaluate fairness
echo ""
echo "[3/7] Running fairness evaluation..."
python training/eval_fairness.py

# Step 4: Generate explainability report
echo ""
echo "[4/7] Generating SHAP explainability report..."
python training/explain.py

# Step 5: Start service, run tests, then stop
echo ""
echo "[5/7] Starting service and running pytest..."
python service/app.py &
SERVICE_PID=$!
sleep 3

# Verify service is up
curl -s http://localhost:8000/health | python -m json.tool
echo "Service is up. Running pytest..."
python -m pytest tests/ -v
echo "All tests passed."

# Step 6: Run load test
echo ""
echo "[6/7] Running load test..."
python scripts/run_load_test.py

# Step 7: Run failure injection demo
echo ""
echo "[7/7] Running failure injection demo..."
python scripts/run_failure_demo.py

# Stop service
kill $SERVICE_PID 2>/dev/null || true
wait $SERVICE_PID 2>/dev/null || true

# Step 8: Regenerate sign-off doc from evidence
echo ""
echo "[8/8] Generating reliability sign-off from evidence..."
python scripts/generate_signoff.py

echo ""
echo "========================================"
echo "ALL STEPS COMPLETED SUCCESSFULLY"
echo "========================================"
echo "Evidence files in results/:"
ls -la results/
echo ""
echo "Sign-off doc: docs/reliability_sign_off.md"
