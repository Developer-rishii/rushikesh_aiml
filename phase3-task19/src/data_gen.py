"""
Stage B.2 / C.2 / D.2 — "Build on real data, not a curated sample."

We don't have PlaceMux's production log warehouse, so this generates a
*realistic logged-interaction dataset* with the same shape, noise, and bias
patterns real hiring-marketplace logs have:
  - impressions -> clicks -> applications -> shortlists (funnel, not i.i.d.)
  - multiple tenants (enterprises) with different hiring bars
  - a protected attribute (gender_proxy) that is NEVER used as a feature but
    IS used to inject realistic historical bias into ground truth, so our
    fairness guardrail in Stage C has something real to catch.
  - noisy skill-overlap, experience, distance, tenant-specific weighting of
    what "good match" means.

This file is the single source of truth for how raw logs are read.
features.py must consume this schema exactly (train/serve consistency).
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

TENANTS = ["acme_bank", "nimbus_tech", "orion_retail"]

SKILLS = ["python", "sql", "react", "aws", "excel", "leadership",
          "sales", "java", "ml", "communication"]


def _gen_candidates(n=4000):
    df = pd.DataFrame({
        "candidate_id": [f"C{i:05d}" for i in range(n)],
        "years_exp": RNG.gamma(3, 2, n).clip(0, 25).round(1),
        "distance_km": RNG.exponential(15, n).clip(0, 200).round(1),
        "num_skills": RNG.integers(2, 8, n),
        # gender_proxy exists ONLY to simulate historical bias in logs.
        # It must never appear in features.py.
        "gender_proxy": RNG.choice(["A", "B"], n, p=[0.55, 0.45]),
    })
    df["skills"] = df["num_skills"].apply(
        lambda k: list(RNG.choice(SKILLS, size=k, replace=False)))
    return df


def _gen_jobs(n=120):
    df = pd.DataFrame({
        "job_id": [f"J{i:04d}" for i in range(n)],
        "tenant_id": RNG.choice(TENANTS, n),
        "req_years_exp": RNG.integers(0, 10, n),
        "num_req_skills": RNG.integers(2, 6, n),
    })
    df["req_skills"] = df["num_req_skills"].apply(
        lambda k: list(RNG.choice(SKILLS, size=k, replace=False)))
    return df


def _skill_overlap(cskills, jskills):
    c, j = set(cskills), set(jskills)
    return len(c & j) / max(1, len(j))


def generate_logs(n_candidates=4000, n_jobs=120, impressions_per_job=60,
                   seed=42):
    """Returns (candidates, jobs, logs) where logs = impression-level rows
    with realistic funnel outcomes (click/apply/shortlist) and a
    historically-biased label used only to demonstrate the fairness
    guardrail — the ranking model is trained WITHOUT gender_proxy."""
    global RNG
    RNG = np.random.default_rng(seed)
    candidates = _gen_candidates(n_candidates)
    jobs = _gen_jobs(n_jobs)

    rows = []
    for _, job in jobs.iterrows():
        shown = candidates.sample(impressions_per_job, replace=False,
                                   random_state=RNG.integers(1e9))
        for _, cand in shown.iterrows():
            overlap = _skill_overlap(cand.skills, job.req_skills)
            exp_gap = cand.years_exp - job.req_years_exp
            exp_fit = 1 / (1 + np.exp(-0.3 * exp_gap))
            dist_fit = np.exp(-cand.distance_km / 40)

            true_quality = (0.5 * overlap + 0.3 * exp_fit + 0.2 * dist_fit)
            # historical bias baked into who actually got shortlisted in
            # the old (pre-model) process — this is what the fairness
            # guardrail must catch if a tenant's config tries to encode it.
            bias_penalty = 0.06 if cand.gender_proxy == "B" else 0.0
            observed_quality = np.clip(
                true_quality - bias_penalty + RNG.normal(0, 0.08), 0, 1)

            clicked = RNG.random() < (0.15 + 0.5 * observed_quality)
            applied = clicked and (RNG.random() < (0.1 + 0.6 * observed_quality))
            shortlisted = applied and (RNG.random() < (0.05 + 0.7 * observed_quality))

            rows.append(dict(
                job_id=job.job_id, tenant_id=job.tenant_id,
                candidate_id=cand.candidate_id,
                skill_overlap=overlap, years_exp=cand.years_exp,
                req_years_exp=job.req_years_exp, distance_km=cand.distance_km,
                num_skills=cand.num_skills,
                gender_proxy=cand.gender_proxy,  # kept ONLY for audit/eval
                clicked=int(clicked), applied=int(applied),
                shortlisted=int(shortlisted),
                label=observed_quality,
            ))
    logs = pd.DataFrame(rows)
    return candidates, jobs, logs


if __name__ == "__main__":
    from config import DATA_DIR
    cands, jobs, logs = generate_logs()
    cands.to_pickle(DATA_DIR / "candidates.pkl")
    jobs.to_pickle(DATA_DIR / "jobs.pkl")
    logs.to_pickle(DATA_DIR / "logs.pkl")
    print(f"candidates={len(cands)} jobs={len(jobs)} impressions={len(logs)}")
    print(logs.groupby("tenant_id")[["clicked", "applied", "shortlisted"]].mean())
