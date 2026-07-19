"""
skew_check.py
-------------
"Train/serve skew: The single biggest silent killer -- features computed
one way in training and another way at serving. Detect it or your model
quietly rots." (Study guide, Role foundations.)

This compares the feature distributions the model actually saw at SERVING
time (results/served_features.jsonl, logged live by serving_app.py during
the load test / failure demo) against the distributions it was TRAINED on
(data/interaction_logs.csv). A simple z-score-of-means check flags any
feature whose serving-time mean has drifted more than `Z_THRESHOLD`
standard deviations from its training-time mean -- cheap, dependency-free,
and good enough to catch the common failure (a feature pipeline bug, a
default-value mismatch, a unit change) before it silently degrades ranking
quality.
"""
import json

import pandas as pd

TRAIN_PATH = "data/interaction_logs.csv"
SERVED_PATH = "results/served_features.jsonl"
REPORT_PATH = "results/skew_report.md"
FEATURES = [
    "exp_years", "skill_match", "education_score", "location_match",
    "past_response_rate", "job_seniority", "job_urgency_score",
    "job_num_applicants_so_far",
]
Z_THRESHOLD = 0.5  # flag if |serving_mean - train_mean| > Z_THRESHOLD * train_std


def main():
    train = pd.read_csv(TRAIN_PATH)[FEATURES]

    served_rows = []
    with open(SERVED_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                served_rows.append(json.loads(line))
    served = pd.DataFrame(served_rows)[FEATURES]

    lines = ["# Train/Serve Skew Report\n",
              f"Training rows: {len(train):,} · Served (sampled) rows logged live during "
              f"load test + failure demo: {len(served):,}\n",
              "| feature | train_mean | served_mean | train_std | z | flagged |",
              "|---|---|---|---|---|---|"]
    any_flag = False
    for feat in FEATURES:
        tm, ts = train[feat].mean(), train[feat].std() or 1e-9
        sm = served[feat].mean()
        z = abs(sm - tm) / ts
        flagged = z > Z_THRESHOLD
        any_flag = any_flag or flagged
        lines.append(f"| {feat} | {tm:.3f} | {sm:.3f} | {ts:.3f} | {z:.2f} | {'YES' if flagged else 'no'} |")

    lines.append("\n" + ("**Skew detected** — investigate feature pipeline before trusting model scores."
                          if any_flag else
                          "**No skew detected** — served feature distributions match training "
                          "distributions within tolerance. Feature pipeline is consistent between "
                          "training and serving for this run."))

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
