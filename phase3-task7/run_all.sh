#!/usr/bin/env bash
set -e

# Detect working python executable
if python.exe -c "import numpy" &> /dev/null; then
    PY_CMD="python.exe"
elif py -3 -c "import numpy" &> /dev/null; then
    PY_CMD="py -3"
elif python3 -c "import numpy" &> /dev/null; then
    PY_CMD="python3"
elif python -c "import numpy" &> /dev/null; then
    PY_CMD="python"
else
    PY_CMD="python.exe"
fi

echo "=========================================================================="
echo "Phase 3 Task 7: Activation & Onboarding Funnel Optimization Pipeline"
echo "Python Command: $PY_CMD"
echo "=========================================================================="

echo "[1/3] Generating synthetic user, job, and interaction log dataset..."
$PY_CMD -m data.generate_synthetic_data

echo ""
echo "[2/3] Executing Model Evaluation, Skew Check & Fairness Audit..."
$PY_CMD -m src.evaluate

echo ""
echo "[3/3] Executing Full Unit Test Suite & Failure Mode Verification..."
$PY_CMD -m pytest tests/ -v

echo ""
echo "=========================================================================="
echo "SUCCESS: All stages executed cleanly with zero errors."
echo "Reproducible empirical evidence generated for Stage E evaluation."
echo "=========================================================================="