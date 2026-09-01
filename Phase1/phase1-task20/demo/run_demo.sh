#!/usr/bin/env bash
# run_demo.sh
# One-command live demo: starts the API, hits health + real predictions +
# an edge case, then leaves the server running for manual poking.
#
# Usage: bash demo/run_demo.sh

set -e
cd "$(dirname "$0")/.."

echo ">> Starting Flask model service..."
(cd src && python3 app.py > ../logs/server_log.txt 2>&1 &)
sleep 3

echo ">> Health check:"
curl -s http://127.0.0.1:5000/health | python3 -m json.tool

echo -e "\n>> Real prediction (malignant-shaped sample):"
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}' \
  | python3 -m json.tool

echo -e "\n>> Real prediction (benign-shaped sample):"
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]}' \
  | python3 -m json.tool

echo -e "\n>> Edge case (garbage input -> validation error, not a crash):"
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3]}' | python3 -m json.tool

echo -e "\n>> Server is running at http://127.0.0.1:5000 — try your own requests."
