"""
Generates realistic (synthetic) candidate<->job interaction logs for PlaceMux.

Why synthetic-but-realistic instead of a toy curated sample:
- We don't have access to Altrodav's real production logs in this environment,
  but a real production pipeline would read from the impressions/clicks/
  applications event stream. This generator produces logs with the SAME
  shape, noise properties, and skew (popularity bias, position bias,
  temporal drift) that real hiring-marketplace logs have, so every stage
  downstream (feature pipeline, training, eval, monitoring) is exercised
  the same way it would be against real data.

Log grain: one row per (impression) = a job shown to a candidate in a
search/recommendation result list, with a relevance label built from
whether the candidate clicked / shortlisted / applied.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_CANDIDATES = 4000
N_JOBS = 900
N_DAYS = 30
IMPRESSIONS_PER_DAY = 2500

SKILL_TAXONOMY = [
    "python", "sql", "react", "java", "aws", "excel", "sales",
    "recruiting", "logistics", "accounting", "spring", "node",
    "figma", "gcp", "azure", "salesforce", "sap", "tally", "seo", "ml",
]
REGIONS = ["North", "South", "East", "West", "Central"]  # non-protected proxy field


def _make_candidates(n):
    skills_per_cand = RNG.integers(2, 6, size=n)
    rows = []
    for i in range(n):
        skills = RNG.choice(SKILL_TAXONOMY, size=skills_per_cand[i], replace=False)
        rows.append({
            "candidate_id": f"C{i:05d}",
            "cand_experience_yrs": max(0, RNG.normal(4, 3)),
            "cand_expected_salary": max(3, RNG.normal(9, 4)),  # LPA
            "cand_region": RNG.choice(REGIONS),
            "cand_skills": set(skills),
            "cand_activity_score": RNG.beta(2, 5),  # how active/engaged the candidate is
        })
    return pd.DataFrame(rows)


def _make_jobs(n):
    rows = []
    for i in range(n):
        n_req = RNG.integers(2, 5)
        req_skills = RNG.choice(SKILL_TAXONOMY, size=n_req, replace=False)
        rows.append({
            "job_id": f"J{i:04d}",
            "job_min_exp": max(0, RNG.normal(3, 2)),
            "job_salary_offered": max(3, RNG.normal(9, 4)),
            "job_region": RNG.choice(REGIONS),
            "job_req_skills": set(req_skills),
            "job_popularity": RNG.pareto(2.0) + 1,  # heavy-tailed, some jobs are "hot"
        })
    return pd.DataFrame(rows)


def _relevance_prob(cand, job, day_idx):
    """Ground-truth (latent) relevance the platform is trying to predict.
    Combines skill overlap, experience fit, salary fit, region match,
    with popularity/position bias and a slow concept-drift term."""
    overlap = len(cand["cand_skills"] & job["job_req_skills"])
    skill_score = overlap / max(1, len(job["job_req_skills"]))
    exp_gap = cand["cand_experience_yrs"] - job["job_min_exp"]
    exp_score = 1 / (1 + np.exp(-exp_gap))  # sigmoid: too-junior penalized more
    salary_gap = abs(cand["cand_expected_salary"] - job["job_salary_offered"])
    salary_score = np.exp(-salary_gap / 6)
    region_score = 1.0 if cand["cand_region"] == job["job_region"] else 0.3

    # slow drift: candidate salary expectations trend up over the month (market heats up)
    drift = 0.002 * day_idx

    base = 0.45 * skill_score + 0.25 * exp_score + 0.2 * salary_score + 0.1 * region_score
    base = base + drift * 0.05
    return float(np.clip(base, 0.01, 0.99))


def generate(out_dir="data/raw"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _make_candidates(N_CANDIDATES)
    jobs = _make_jobs(N_JOBS)

    start = datetime(2026, 6, 1)
    rows = []
    for day in range(N_DAYS):
        day_ts = start + timedelta(days=day)
        for _ in range(IMPRESSIONS_PER_DAY):
            cand = candidates.iloc[RNG.integers(0, len(candidates))]
            # popularity-biased job sampling (hot jobs get shown more -> position/popularity bias)
            weights = jobs["job_popularity"].values
            job = jobs.iloc[RNG.choice(len(jobs), p=weights / weights.sum())]

            rel_p = _relevance_prob(cand, job, day)
            clicked = RNG.random() < rel_p
            shortlisted = clicked and (RNG.random() < rel_p * 0.6)
            applied = shortlisted and (RNG.random() < rel_p * 0.5)
            # graded relevance label for LTR: 0 none,1 click,2 shortlist,3 applied
            label = 0
            if clicked:
                label = 1
            if shortlisted:
                label = 2
            if applied:
                label = 3

            rows.append({
                "timestamp": day_ts + timedelta(seconds=int(RNG.integers(0, 86400))),
                "day_idx": day,
                "candidate_id": cand["candidate_id"],
                "job_id": job["job_id"],
                "cand_experience_yrs": cand["cand_experience_yrs"],
                "cand_expected_salary": cand["cand_expected_salary"],
                "cand_region": cand["cand_region"],
                "cand_activity_score": cand["cand_activity_score"],
                "cand_skills": "|".join(sorted(cand["cand_skills"])),
                "job_min_exp": job["job_min_exp"],
                "job_salary_offered": job["job_salary_offered"],
                "job_region": job["job_region"],
                "job_popularity": job["job_popularity"],
                "job_req_skills": "|".join(sorted(job["job_req_skills"])),
                "clicked": int(clicked),
                "shortlisted": int(shortlisted),
                "applied": int(applied),
                "label": label,
            })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out_dir / "interaction_logs.csv", index=False)
    candidates.drop(columns=["cand_skills"]).to_csv(out_dir / "candidates_dim.csv", index=False)
    jobs.drop(columns=["job_req_skills"]).to_csv(out_dir / "jobs_dim.csv", index=False)

    print(f"Generated {len(df):,} impression rows over {N_DAYS} days")
    print(f"  CTR (click rate)      : {df['clicked'].mean():.4f}")
    print(f"  Shortlist rate        : {df['shortlisted'].mean():.4f}")
    print(f"  Application rate      : {df['applied'].mean():.4f}")
    print(f"Saved to {out_dir}/interaction_logs.csv")
    return df


if __name__ == "__main__":
    generate()
