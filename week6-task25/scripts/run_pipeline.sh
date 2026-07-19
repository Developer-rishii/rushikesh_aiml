#!/usr/bin/env bash
# run_pipeline.sh
# Reproduces the entire Task 25 pipeline from a clean checkout, in order:
#   1. install dependencies
#   2. generate historical + simulated production data
#   3. train the model and validate against the baseline
#   4. run the live monitoring simulation end-to-end (writes evidence/)
#   5. run the automated test suite
set -e

cd "$(dirname "$0")/.."

echo "== [1/5] Installing dependencies =="
pip install -r requirements.txt --break-system-packages -q || pip install -r requirements.txt -q

echo "== [2/5] Generating historical + production data =="
python3 -m src.data_generator

echo "== [3/5] Training model + baseline =="
python3 -m src.train_model

echo "== [4/5] Running live monitoring simulation (produces evidence/) =="
python3 scripts/simulate_and_monitor.py

echo "== [5/5] Running test suite =="
python3 -m unittest discover tests -v

echo ""
echo "Pipeline complete. See evidence/ for metrics_report.json, demo_walkthrough.md, plots/, run_logs.txt."
echo "To serve the live API: uvicorn src.api.main:app --reload --port 8000"
