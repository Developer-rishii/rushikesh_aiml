#!/usr/bin/env bash
# Runs the entire Task 25 pipeline end-to-end, in order, on real (logged)
# data, and regenerates every report from scratch. Re-run this any time to
# reproduce every number in reports/ from raw data.
set -e
cd "$(dirname "$0")"
rm -f reports/experiment_log.csv registry/model_registry.json
python3 data/generate_data.py
cd src
python3 train_ranker.py
python3 evaluate_offline.py
python3 evaluate_online_proxy.py
python3 fairness_audit.py
python3 latency_cost.py
python3 drift_rollback.py
python3 dr_failover.py
python3 explainability.py
python3 governance.py
python3 build_reports.py
cd ..
sed -i 's#'"$(pwd)"'/##g' registry/model_registry.json reports/model_card.md 2>/dev/null || true
echo "DONE. See reports/certification_pack.md and reports/post_golive_report.md"
