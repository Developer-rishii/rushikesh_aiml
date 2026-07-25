"""
skew_check.py
--------------
"The single biggest silent killer: features computed one way in training
and another way at serving." (study guide, section 5)

Takes ONE snapshot of raw candidate rows and runs it through both
compute_training_features() and compute_serving_features() from features.py,
then diffs every feature column. This is meant to actually catch the
recency_days bug seeded in features.py, not just assert skew doesn't exist.
"""

import json

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, compute_serving_features, compute_training_features


def run_skew_check():
    hist = pd.read_csv("data/historical_logs.csv")
    as_of_day = int(hist["posted_day"].max())

    train_feats = compute_training_features(hist, as_of_day).sort_values(["query_id", "candidate_id"]).reset_index(drop=True)
    serve_feats = compute_serving_features(hist, as_of_day).sort_values(["query_id", "candidate_id"]).reset_index(drop=True)

    report = {"as_of_day": as_of_day, "n_rows_compared": int(len(train_feats)), "columns": {}}
    any_skew = False
    for col in FEATURE_COLUMNS:
        diff = (train_feats[col] - serve_feats[col]).to_numpy()
        mismatched = int(np.sum(np.abs(diff) > 1e-9))
        pct_mismatched = mismatched / len(train_feats) * 100
        col_report = {
            "mismatched_rows": mismatched,
            "pct_mismatched": round(pct_mismatched, 2),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
        }
        report["columns"][col] = col_report
        if mismatched > 0:
            any_skew = True

    report["skew_detected"] = any_skew
    report["verdict"] = (
        "FAIL: train/serve skew detected in one or more features. Root-caused to "
        "'recency_days' — the serving path applies a +1 day offset the training "
        "path does not (see features.py comment). This must be fixed before "
        "shipping treatment; a model trained on correct recency but served with "
        "skewed recency will silently degrade in production."
        if any_skew else
        "PASS: no train/serve skew detected."
    )
    with open("artifacts/skew_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    report = run_skew_check()
    print(json.dumps(report, indent=2))
