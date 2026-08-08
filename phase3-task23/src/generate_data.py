"""
Stage B.2 — Build on real (logged) data.
Since this is a training/study-guide task with no access to PlaceMux's actual
production DB, this script generates a REALISTIC LOGGED-INTERACTION dataset:
candidates, jobs, and timestamped impression/click/application/shortlist events,
with a protected attribute (for fairness auditing) and deliberate train/serve
skew (a feature computed differently at "training" time vs "serving" time).
This limitation is stated explicitly in the audit pack (lineage.json) so the
audit trail is honest about data provenance — required by Pitfall #3/#4.
"""
import numpy as np
import pandas as pd
import json, hashlib, os, sys
from datetime import datetime, timedelta, timezone

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
rng = np.random.default_rng(seed)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

N_CANDIDATES = 4000
N_JOBS = 150
N_IMPRESSIONS = 60000

# ---- Candidates ----
protected_group = rng.choice(["A", "B"], size=N_CANDIDATES, p=[0.55, 0.45])  # synthetic protected attribute
candidates = pd.DataFrame({
    "candidate_id": [f"C{i:05d}" for i in range(N_CANDIDATES)],
    "years_experience": np.clip(rng.normal(4, 3, N_CANDIDATES), 0, 25).round(1),
    "skill_match_score": np.clip(rng.normal(0.6, 0.2, N_CANDIDATES), 0, 1).round(3),
    "profile_completeness": np.clip(rng.normal(0.75, 0.15, N_CANDIDATES), 0, 1).round(3),
    "protected_group": protected_group,          # NEVER used as a model feature; used only for fairness audit
    "consent_ts": [
        (datetime(2025, 1, 1) + timedelta(days=int(d))).isoformat()
        for d in rng.integers(0, 500, N_CANDIDATES)
    ],
})

# ---- Jobs ----
jobs = pd.DataFrame({
    "job_id": [f"J{i:04d}" for i in range(N_JOBS)],
    "seniority_level": rng.integers(1, 5, N_JOBS),
    "req_skill_score": np.clip(rng.normal(0.6, 0.15, N_JOBS), 0, 1).round(3),
})

# ---- Interaction logs (impressions -> click -> application -> shortlist) ----
cand_idx = rng.integers(0, N_CANDIDATES, N_IMPRESSIONS)
job_idx = rng.integers(0, N_JOBS, N_IMPRESSIONS)

c = candidates.iloc[cand_idx].reset_index(drop=True)
j = jobs.iloc[job_idx].reset_index(drop=True)

# TRUE relevance signal driving real behaviour (ground truth, NOT observed by model)
match = 1 - np.abs(c["skill_match_score"].values - j["req_skill_score"].values)
true_relevance = (0.5 * match + 0.3 * (c["years_experience"].values / 25) +
                   0.2 * c["profile_completeness"].values)
true_relevance = (true_relevance - true_relevance.min()) / (true_relevance.max() - true_relevance.min())

# CALIBRATION TO REAL-WORLD BENCHMARKS (e.g. LinkedIn / typical job board):
# CTR benchmark: ~8%.  Apply rate: ~20% of clicks. Shortlist rate: ~15% of applications.
# We scale true_relevance to hit these approximate macroscopic averages.
click_p = np.clip(true_relevance * 0.16, 0, 1) # Avg true_rel is ~0.5, so avg CTR ~8%
clicked = rng.random(N_IMPRESSIONS) < click_p
applied = clicked & (rng.random(N_IMPRESSIONS) < 0.20) # 20% apply rate from clicks
shortlisted = applied & (rng.random(N_IMPRESSIONS) < (true_relevance * 0.3)) # ~15% of applies

# TRAIN/SERVE SKEW (deliberate, to be caught by drift_monitor.py):
# training-time feature = recency computed in DAYS at batch time (correct)
# serving-time feature  = recency computed in HOURS mistakenly cast to same column name
event_ts = pd.to_datetime("2026-06-01") - pd.to_timedelta(rng.integers(0, 400, N_IMPRESSIONS), unit="D")
delta = pd.to_datetime("2026-06-01") - event_ts
recency_days_TRAIN = delta.days
recency_HOURS_SERVE_BUG = delta.total_seconds() / 3600.0

interactions = pd.DataFrame({
    "impression_id": [f"I{i:07d}" for i in range(N_IMPRESSIONS)],
    "candidate_id": c["candidate_id"].values,
    "job_id": j["job_id"].values,
    "years_experience": c["years_experience"].values,
    "skill_match_score": c["skill_match_score"].values,
    "profile_completeness": c["profile_completeness"].values,
    "seniority_level": j["seniority_level"].values,
    "req_skill_score": j["req_skill_score"].values,
    "recency_feature_train": recency_days_TRAIN,   # used at training time
    "recency_feature_serve": recency_HOURS_SERVE_BUG,  # what serving would compute (skewed)
    "event_ts": event_ts.astype(str),
    "clicked": clicked.astype(int),
    "applied": applied.astype(int),
    "shortlisted": shortlisted.astype(int),
    "true_relevance": true_relevance.round(4),  # kept only for offline eval / audit, not a feature
})

candidates.to_csv(f"{OUT}/candidates.csv", index=False)
jobs.to_csv(f"{OUT}/jobs.csv", index=False)
interactions.to_csv(f"{OUT}/interactions.csv", index=False)

# Generate NOISY validation slice (real-shaped messiness)
noisy_interactions = interactions.sample(frac=0.1, random_state=seed).copy()
# Introduce missing fields
missing_mask = rng.random(len(noisy_interactions)) < 0.05
noisy_interactions.loc[missing_mask, "years_experience"] = np.nan
# Introduce late-arriving / malformed timestamps
late_mask = rng.random(len(noisy_interactions)) < 0.02
noisy_interactions.loc[late_mask, "event_ts"] = "1970-01-01 00:00:00"
# Duplicate some impressions
duplicates = noisy_interactions.sample(frac=0.05, random_state=seed)
noisy_interactions = pd.concat([noisy_interactions, duplicates])
noisy_interactions.to_csv(f"{OUT}/interactions_noisy.csv", index=False)

# Data lineage hash (evidence, not claim)
h = hashlib.sha256(pd.util.hash_pandas_object(interactions).values.tobytes()).hexdigest()[:16]
with open(f"{OUT}/data_manifest.json", "w") as f:
    json.dump({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": N_CANDIDATES, "n_jobs": N_JOBS, "n_impressions": N_IMPRESSIONS,
        "dataset_sha256_16": h,
        "provenance_note": "Synthetic logged-interaction dataset (no access to PlaceMux prod DB in this "
                            "study-guide context). Statistically realistic funnel: impression->click->"
                            "apply->shortlist, with an injected train/serve skew bug for detection. "
                            "Calibrated to industry benchmarks: ~8% CTR, ~20% apply rate, ~15% shortlist rate. "
                            "Includes a noisy validation slice (interactions_noisy.csv) with missing data, "
                            "duplicate rows, and malformed timestamps to test pipeline robustness.",
    }, f, indent=2)

print(f"Generated {len(candidates)} candidates, {len(jobs)} jobs, {len(interactions)} interaction logs.")
print(f"Dataset hash: {h}")
