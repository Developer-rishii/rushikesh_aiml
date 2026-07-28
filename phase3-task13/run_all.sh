#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "== 1/3 generating data =="
python3 data/generate_data.py
echo "== 2/3 running pipeline (Stages B-E, writes outputs/experiment_log.json) =="
python3 -m src.pipeline
echo "== 3/3 running live demo =="
python3 demo.py
