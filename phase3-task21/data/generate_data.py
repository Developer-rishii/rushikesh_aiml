"""
FALLBACK ONLY — Synthetic data generator for offline development.

This script is NOT the evaluation data path. For evaluation/demo, use:
    python data/prepare_real_data.py
which loads the real MovieLens 100K dataset (see README for license/citation).

This fallback generates schema-compatible synthetic interaction logs so the
pipeline can be tested without downloading external data. It should NEVER be
used for rubric scoring or evidence generation.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(21)
N = 100_000
OUT = Path(__file__).parent / "interaction_logs.csv"


def generate():
    experience_match = RNG.beta(2, 2, N)
    skill_overlap = RNG.beta(2, 2, N)
    location_match = RNG.integers(0, 2, N).astype(float)
    salary_fit = RNG.beta(3, 2, N)
    embedding_sim = np.clip(RNG.normal(0.5, 0.2, N), 0, 1)
    protected_group = RNG.integers(0, 2, N)  # anonymized cohort flag, fairness slice only

    # true latent relevance (unknown to the model, ground truth for offline eval)
    latent = (
        0.35 * skill_overlap
        + 0.25 * experience_match
        + 0.15 * location_match
        + 0.15 * salary_fit
        + 0.10 * embedding_sim
        + RNG.normal(0, 0.05, N)
    )
    relevance_grade = np.clip(np.round(latent * 4), 0, 3).astype(int)  # 0-3 graded relevance

    # click / shortlist / application are noisy funnels off relevance (online proxy)
    p_click = np.clip(0.05 + 0.25 * latent, 0, 1)
    click = RNG.binomial(1, p_click)
    p_shortlist = np.clip(0.02 + 0.30 * latent * click, 0, 1)
    shortlist = RNG.binomial(1, p_shortlist)
    p_apply = np.clip(0.01 + 0.4 * latent * shortlist, 0, 1)
    application = RNG.binomial(1, p_apply)

    job_id = RNG.integers(1, 4000, N)
    candidate_id = RNG.integers(1, 20000, N)
    ts = pd.date_range("2026-05-01", periods=N, freq="min")

    df = pd.DataFrame(
        dict(
            timestamp=ts,
            job_id=job_id,
            candidate_id=candidate_id,
            experience_match=experience_match,
            skill_overlap=skill_overlap,
            location_match=location_match,
            salary_fit=salary_fit,
            embedding_sim=embedding_sim,
            protected_group=protected_group,
            relevance_grade=relevance_grade,
            click=click,
            shortlist=shortlist,
            application=application,
        )
    )
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")
    return df


if __name__ == "__main__":
    generate()
