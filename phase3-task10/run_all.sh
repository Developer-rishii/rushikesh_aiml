#!/usr/bin/env bash
# Runs Stages A-E of the study guide's build pipeline, in order.
# The order below is not cosmetic: pre_registration.py MUST complete before
# ab_simulation.py runs, or readout.py's no-peeking guard will refuse to run.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "== Stage A/B: simulate historical logs (stand-in for real production logs) =="
python3 src/data_simulation.py

echo "== Stage B: train baseline + treatment, log to model registry =="
python3 src/train_ranker.py

echo "== Stage B: offline evaluation on held-out queries =="
python3 src/offline_eval.py

echo "== Stage B/C: PRE-REGISTER hypothesis + metric (locked before online data exists) =="
python3 src/pre_registration.py

echo "== Stage B/C: run the online A/B for the pre-registered 14-day window =="
python3 src/ab_simulation.py

echo "== Stage C: honest readout (effect size, significance, guardrails) =="
python3 src/readout.py

echo "== Stage D: ship / do-not-ship decision =="
python3 src/decision.py

echo "== Stage E: dependency/failure/edge-case checks =="
python3 src/skew_check.py
python3 src/failure_test.py

echo "== Explainability worked example =="
python3 src/explain.py

echo "== Evidence plots =="
python3 src/make_plots.py

echo ""
echo "DONE. All artifacts written to artifacts/. See EXPERIMENT_LOG.md for the narrative readout."
