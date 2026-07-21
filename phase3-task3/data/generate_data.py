"""
Generates a realistic candidate<->job interaction log.

IMPORTANT / HONESTY NOTE (see README, section "Data caveat"):
We do not have access to PlaceMux's real production logs, so this script
produces a *simulated* but structurally realistic stand-in: impressions,
clicks and applications, with the same feature set, noise structure, class
imbalance and a deliberately injected relationship between features and
outcome (so that a ranking model can genuinely learn something and offline
metrics are meaningful, not just fitting noise). Every number reported later
is computed from this generated file, not fabricated after the fact -- the
generation happens once, is saved to disk, and every downstream script reads
that same frozen CSV so results are reproducible.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import INTERACTIONS_CSV, RANDOM_SEED, PROTECTED_GROUP_COL, LABEL_COL

N_JOBS = 800
CANDS_PER_JOB = 40  # impressions shown per job posting (candidate shortlisted pool)


def generate(n_jobs=N_JOBS, cands_per_job=CANDS_PER_JOB, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    job_ids = [f"job_{i:05d}" for i in range(n_jobs)]

    for job_id in job_ids:
        job_urgency = rng.beta(2, 3)          # 0..1, most jobs not desperate
        recruiter_rating = rng.normal(3.8, 0.6)
        recruiter_rating = float(np.clip(recruiter_rating, 1, 5))

        n_cand = cands_per_job
        skill_overlap = rng.beta(2, 2, n_cand)              # 0..1
        years_exp = rng.gamma(3, 2, n_cand)                  # skewed, 0..~20
        distance_km = rng.exponential(15, n_cand)
        salary_gap_pct = rng.normal(0, 15, n_cand)           # candidate ask vs budget
        activity_score = rng.beta(2, 3, n_cand)
        response_rate = rng.beta(2, 2, n_cand)
        group = rng.choice(["A", "B"], size=n_cand, p=[0.55, 0.45])

        # True latent relevance driven mostly by skill overlap & experience,
        # softened by distance and salary mismatch -- this is the signal a
        # ranking model should recover. Deliberately NOT dependent on `group`,
        # so a fairness gap in scores later would flag a genuine model bug.
        latent = (
            2.6 * skill_overlap
            + 0.05 * np.clip(years_exp, 0, 12)
            - 0.01 * distance_km
            - 0.01 * np.abs(salary_gap_pct)
            + 0.8 * activity_score
            + 0.5 * response_rate
            + 0.3 * job_urgency
            + rng.normal(0, 0.5, n_cand)  # noise
        )
        # Convert latent score into 3-level relevance label matching the
        # funnel: applied(2) > clicked(1) > no action(0)
        p_apply = 1 / (1 + np.exp(-(latent - 3.2)))
        p_click = 1 / (1 + np.exp(-(latent - 1.6)))
        u = rng.uniform(size=n_cand)
        relevance = np.where(u < p_apply, 2, np.where(u < p_click, 1, 0))

        for i in range(n_cand):
            rows.append({
                "job_id": job_id,
                "candidate_id": f"{job_id}_c{i:03d}",
                "skill_overlap": skill_overlap[i],
                "years_experience": years_exp[i],
                "distance_km": distance_km[i],
                "salary_gap_pct": salary_gap_pct[i],
                "candidate_activity_score": activity_score[i],
                "candidate_past_response_rate": response_rate[i],
                "job_fill_urgency": job_urgency,
                "recruiter_rating": recruiter_rating,
                PROTECTED_GROUP_COL: group[i],
                LABEL_COL: int(relevance[i]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(INTERACTIONS_CSV, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Wrote {len(df)} rows ({df['job_id'].nunique()} jobs) to {INTERACTIONS_CSV}")
    print(df[LABEL_COL].value_counts(normalize=True).round(3))
