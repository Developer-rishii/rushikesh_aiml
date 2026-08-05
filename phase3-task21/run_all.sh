#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || pip install -q -r requirements.txt
python data/prepare_real_data.py   # real MovieLens 100K dataset (primary path)
python src/pipeline.py
python src/demo.py
