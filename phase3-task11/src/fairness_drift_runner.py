"""
fairness_drift_runner.py
========================
Stage 4/5 runner: computes selection-rate parity and feature drift (PSI),
writes reports/fairness_drift.json.  Extracted from the inline heredoc in
run_all.sh so both run_all.sh and run_all.ps1 can call it portably.
"""
import sys, json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fairness_drift import selection_rate_parity, drift_report
from features import FEATURE_COLUMNS
import evaluate as ev

df = pd.read_csv(ROOT / "data" / "raw_logs.csv")
train_df, val_df, test_df = ev.split_by_job(df)
test_scored = pd.read_csv(ROOT / "artifacts" / "test_scored.csv")

report = {
    "selection_rate_parity_top5": {
        "heuristic": selection_rate_parity(test_scored, "score_heuristic"),
        "chosen_model": selection_rate_parity(test_scored, "score_pairwise_corrected"),
    },
    "feature_drift_psi_train_vs_test": drift_report(train_df, test_df, FEATURE_COLUMNS),
}
with open(ROOT / "reports" / "fairness_drift.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
