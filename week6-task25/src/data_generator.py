"""
data_generator.py

Upstream dependency note (Section 3 / Section 10 of the study guide):
    "Waiting on: Production traffic."
Production traffic had not landed by the time this task started. Rather than
block, we generate a real-shaped synthetic stream whose feature distributions
and hire-rate base rates are calibrated to match what PlaceMux's earlier
tasks (skill verification, interview eval, resume parsing) actually produce,
and we log this decision explicitly (see docs/handoff.md) so the founder
knows exactly which numbers are simulated-but-real-shaped vs from live users,
and can swap PROD_TRAFFIC_PATH for a real traffic dump with zero code changes
once it arrives (the schema is identical).

This module produces two datasets:
  1. Historical matches (for training + baseline reference distribution).
  2. A "live" production stream, split into time-ordered batches, with a
     deliberate distribution shift injected partway through (simulating a
     JD-mix change / new segment of employers) so the monitoring system has
     something real to detect and is not just measuring a static, hand-tuned
     demo set.
"""

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_SEED,
    TRAIN_HISTORY_PATH, PROD_TRAFFIC_PATH, LABEL_ARRIVAL_RATE,
)

SEGMENTS = ["software_eng", "data_analyst", "qa_eng", "support_ops", "sales_sdr"]


def _sample_features(n, rng, drift_shift=0.0, segment_bias=None):
    """Sample one batch of feature rows.

    drift_shift > 0 nudges the population towards weaker signals (simulating
    a lower-quality applicant mix / stale skill-verification pipeline) so we
    can prove the monitor actually catches degradation, not just re-plot
    training-time numbers.
    """
    skill_overlap = np.clip(rng.beta(6 - 2 * drift_shift, 3 + 2 * drift_shift, n), 0, 1)
    years_experience = np.clip(rng.gamma(shape=2.2, scale=1.6, size=n) - drift_shift, 0, 20)
    required_years = np.clip(rng.gamma(shape=2.0, scale=1.5, size=n), 0, 15)
    experience_gap = required_years - years_experience
    resume_parse_confidence = np.clip(rng.beta(8, 1.5, n) - 0.05 * drift_shift, 0, 1)
    interview_eval_score = np.clip(rng.beta(5 - 1.5 * drift_shift, 3 + 1.5 * drift_shift, n), 0, 1)
    communication_score = np.clip(rng.beta(5, 3, n) - 0.05 * drift_shift, 0, 1)

    if segment_bias is None:
        segments = rng.choice(SEGMENTS, size=n)
    else:
        segments = rng.choice(SEGMENTS, size=n, p=segment_bias)

    base_rate_map = {"software_eng": 0.30, "data_analyst": 0.24, "qa_eng": 0.20,
                      "support_ops": 0.18, "sales_sdr": 0.22}
    role_historical_hire_rate = np.array([base_rate_map[s] for s in segments]) - 0.03 * drift_shift

    df = pd.DataFrame({
        "skill_overlap_score": skill_overlap,
        "years_experience": years_experience,
        "experience_gap": experience_gap,
        "resume_parse_confidence": resume_parse_confidence,
        "interview_eval_score": interview_eval_score,
        "communication_score": communication_score,
        "role_historical_hire_rate": np.clip(role_historical_hire_rate, 0.02, 0.95),
        "segment": segments,
    })
    return df


def _simulate_outcome(df, rng, noise=0.35):
    """Ground-truth label: was this actually a successful match (hired and
    still in-role / positively reviewed at 90 days)? Built as a logistic
    function of the real signals plus irreducible noise - real hiring
    outcomes are never perfectly predictable from features alone.
    """
    z = (
        3.1 * df["skill_overlap_score"]
        + 0.9 * df["interview_eval_score"]
        + 0.7 * df["communication_score"]
        + 0.55 * df["resume_parse_confidence"]
        - 0.18 * df["experience_gap"].clip(lower=0)
        + 2.0 * df["role_historical_hire_rate"]
        - 5.20
    )
    z = z + rng.normal(0, noise, size=len(df))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    return y


def generate_historical_dataset(n=6000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    df = _sample_features(n, rng, drift_shift=0.0)
    df[TARGET_COLUMN] = _simulate_outcome(df, rng)
    df.insert(0, "match_id", [f"H{ i:06d}" for i in range(n)])
    df.insert(1, "student_id", rng.integers(100000, 999999, n))
    df.insert(2, "job_id", rng.integers(20000, 29999, n))
    df["event_time"] = pd.date_range("2026-04-01", periods=n, freq="min")
    ordered = ["match_id", "student_id", "job_id", "segment", "event_time"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    return df[ordered]


def generate_production_stream(n=2400, n_batches=12, seed=RANDOM_SEED + 1):
    """Simulate a live stream arriving in n_batches ordered windows.

    Batches 0..6  -> distribution matches training (healthy production).
    Batches 7..9  -> mild drift begins (early warning territory).
    Batches 10..11 -> pronounced drift + segment-mix shift (should trip
                      the critical PSI threshold and the metric-degradation
                      alert), simulating e.g. a bulk import of a new, lower
                      verification-quality employer segment.

    Labels: only LABEL_ARRIVAL_RATE fraction of rows get a ground-truth
    outcome by "now" (hiring decisions lag), the rest are legitimately
    pending - the monitor must handle partially-labeled windows, not assume
    every prediction has a known outcome yet.
    """
    rng = np.random.default_rng(seed)
    per_batch = n // n_batches
    frames = []
    start_time = pd.Timestamp("2026-07-01 09:00:00")

    for b in range(n_batches):
        if b <= 6:
            drift = 0.0
            seg_bias = None
        elif b <= 9:
            drift = (b - 6) * 0.5   # 0.5, 1.0, 1.5 -> mild
            seg_bias = [0.16, 0.20, 0.20, 0.24, 0.20]
        else:
            drift = 2.2 + (b - 10) * 0.4  # pronounced
            seg_bias = [0.10, 0.14, 0.18, 0.36, 0.22]

        batch_df = _sample_features(per_batch, rng, drift_shift=drift, segment_bias=seg_bias)
        y_true = _simulate_outcome(batch_df, rng)

        label_known = rng.random(per_batch) < LABEL_ARRIVAL_RATE
        y_true_masked = np.where(label_known, y_true, -1)  # -1 = outcome not yet known

        batch_df["is_successful_match"] = y_true_masked
        batch_df["batch_id"] = b
        batch_df["event_time"] = pd.date_range(start_time + pd.Timedelta(minutes=45 * b), periods=per_batch, freq="s")
        frames.append(batch_df)

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "match_id", [f"P{i:06d}" for i in range(len(df))])
    df.insert(1, "student_id", rng.integers(100000, 999999, len(df)))
    df.insert(2, "job_id", rng.integers(30000, 39999, len(df)))
    ordered = ["match_id", "student_id", "job_id", "segment", "batch_id", "event_time"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    return df[ordered]


def main():
    hist = generate_historical_dataset()
    prod = generate_production_stream()
    hist.to_csv(TRAIN_HISTORY_PATH, index=False)
    prod.to_csv(PROD_TRAFFIC_PATH, index=False)
    print(f"Historical matches:   {hist.shape} -> {TRAIN_HISTORY_PATH}")
    print(f"Production stream:    {prod.shape} -> {PROD_TRAFFIC_PATH}")
    print(f"Historical hire rate: {hist[TARGET_COLUMN].mean():.3f}")
    known = prod[prod[TARGET_COLUMN] != -1]
    print(f"Prod labeled so far:  {len(known)}/{len(prod)} ({len(known)/len(prod):.1%}), hire rate among labeled: {known[TARGET_COLUMN].mean():.3f}")


if __name__ == "__main__":
    main()
