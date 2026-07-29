"""
generate_data.py
=================
Simulates PlaceMux candidate<->job matching interaction logs.

Why simulated: we don't have production DB access in this environment, but
the pipeline below is written so that swapping this file for a real
"SELECT * FROM impressions JOIN clicks JOIN applications" export changes
NOTHING downstream. Every other module only depends on the schema, not on
how the CSV was produced.

Schema (one row = one impression of a candidate against a job):
  event_id, ts, candidate_id, job_id, region, gender,
  skill_match_score, embedding_similarity, experience_years,
  location_match, recruiter_response_rate, past_ctr,
  clicked, applied, shortlisted   (labels)

Drift is injected deliberately after day 120 of a 180-day window:
  - embedding_similarity distribution shifts (new embedding model rollout)
  - recruiter_response_rate degrades market-wide (seasonal effect)
  - the true relationship between skill_match_score and `shortlisted`
    weakens (concept drift: recruiters start weighting soft-skills more)
This lets Stage C (drift detection) be demonstrated on real signal instead
of being asserted with no evidence.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_DAYS = 180
DRIFT_START_DAY = 120
ROWS_PER_DAY = 220


def _make_day(day_idx: int) -> pd.DataFrame:
    n = ROWS_PER_DAY
    drifted = day_idx >= DRIFT_START_DAY
    region = RNG.choice(["North", "South", "West", "East"], size=n, p=[0.3, 0.25, 0.25, 0.2])
    gender = RNG.choice(["M", "F"], size=n, p=[0.58, 0.42])  # protected attribute, for fairness audit

    experience_years = np.clip(RNG.exponential(3.0, n), 0, 20)
    skill_match_score = np.clip(RNG.normal(0.62, 0.18, n), 0, 1)

    if not drifted:
        embedding_similarity = np.clip(RNG.normal(0.55, 0.15, n), 0, 1)
        recruiter_response_rate = np.clip(RNG.normal(0.40, 0.12, n), 0, 1)
    else:
        # embedding model swap shifts the similarity distribution (covariate drift)
        embedding_similarity = np.clip(RNG.normal(0.72, 0.10, n), 0, 1)
        # market-wide recruiter fatigue (covariate drift)
        recruiter_response_rate = np.clip(RNG.normal(0.24, 0.10, n), 0, 1)

    location_match = RNG.choice([0, 1], size=n, p=[0.35, 0.65])
    past_ctr = np.clip(RNG.normal(0.12, 0.05, n), 0, 1)

    # ---- true latent "fit" score that generates labels (ground truth process) ----
    if not drifted:
        latent = (
            1.8 * skill_match_score
            + 1.1 * embedding_similarity
            + 0.6 * location_match
            + 0.4 * (experience_years / 20)
            + 0.5 * recruiter_response_rate
        )
    else:
        # concept drift: skill_match_score matters LESS, soft signals matter MORE
        latent = (
            0.9 * skill_match_score
            + 1.1 * embedding_similarity
            + 0.6 * location_match
            + 0.9 * (experience_years / 20)
            + 0.5 * recruiter_response_rate
        )

    latent += RNG.normal(0, 0.35, n)
    p_click = 1 / (1 + np.exp(-(latent - 2.0)))
    clicked = RNG.binomial(1, np.clip(p_click, 0.02, 0.95))

    p_apply = np.clip(p_click * 0.55 + 0.05, 0, 0.9)
    applied = clicked & RNG.binomial(1, p_apply)

    p_shortlist = np.clip(p_apply * 0.5 + 0.03, 0, 0.85)
    shortlisted = applied & RNG.binomial(1, p_shortlist)

    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(day_idx, unit="D")
    df = pd.DataFrame({
        "event_id": [f"d{day_idx}_{i}" for i in range(n)],
        "ts": ts,
        "day_idx": day_idx,
        "candidate_id": RNG.integers(10_000, 99_999, n),
        "job_id": RNG.integers(1_000, 9_999, n),
        "region": region,
        "gender": gender,
        "skill_match_score": skill_match_score,
        "embedding_similarity": embedding_similarity,
        "experience_years": experience_years,
        "location_match": location_match,
        "recruiter_response_rate": recruiter_response_rate,
        "past_ctr": past_ctr,
        "clicked": clicked.astype(int),
        "applied": applied.astype(int),
        "shortlisted": shortlisted.astype(int),
    })
    return df


def generate(path: str = "data/raw_logs.csv") -> pd.DataFrame:
    days = [_make_day(d) for d in range(N_DAYS)]
    df = pd.concat(days, ignore_index=True)
    df.to_csv(path, index=False)
    print(f"[generate_data] wrote {len(df):,} rows -> {path}")
    print(f"[generate_data] pre-drift days=0..{DRIFT_START_DAY-1}, drift injected from day {DRIFT_START_DAY}")
    return df


if __name__ == "__main__":
    generate()
