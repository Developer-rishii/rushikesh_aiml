"""
Stage B.2 / C.2 / D.2 - "Build it on real data"
Generates a logged interaction dataset that stands in for PlaceMux's real
impression/click/application/shortlist logs. Not curated: includes noise,
skew, a drifting period (for monitoring) and a protected-group proxy
(group A/B) for fairness auditing, per DPDP / fairness constraints.
Every run is seeded and written to disk so results are reproducible.
"""
import numpy as np
import pandas as pd
import os

SEED = 25
rng = np.random.default_rng(SEED)
OUT = os.path.dirname(__file__)

N_CANDIDATES = 4000
N_JOBS = 300
N_IMPRESSIONS = 60000

# --- candidates -------------------------------------------------------
cand_ids = np.arange(N_CANDIDATES)
cand_skill = rng.normal(0, 1, N_CANDIDATES)          # latent skill score
cand_exp_years = rng.integers(0, 15, N_CANDIDATES)
# protected-group proxy (e.g. gender/region proxy) - NOT used as a model
# feature, only for the fairness audit, as required by the study guide.
cand_group = rng.choice(["A", "B"], N_CANDIDATES, p=[0.55, 0.45])
candidates = pd.DataFrame({
    "candidate_id": cand_ids,
    "skill_score": cand_skill,
    "exp_years": cand_exp_years,
    "group": cand_group,
})

# --- jobs ---------------------------------------------------------------
job_ids = np.arange(N_JOBS)
job_seniority = rng.integers(0, 15, N_JOBS)
job_comp = rng.normal(0, 1, N_JOBS)
jobs = pd.DataFrame({
    "job_id": job_ids,
    "job_seniority": job_seniority,
    "job_comp_level": job_comp,
})

# --- impressions / interactions -----------------------------------------
imp_cand = rng.choice(cand_ids, N_IMPRESSIONS)
imp_job = rng.choice(job_ids, N_IMPRESSIONS)
# a synthetic "day" axis, days 0..29, used later to simulate a staged rollout
imp_day = rng.integers(0, 30, N_IMPRESSIONS)

df = pd.DataFrame({"candidate_id": imp_cand, "job_id": imp_job, "day": imp_day})
df = df.merge(candidates, on="candidate_id").merge(jobs, on="job_id")

# true relevance is a function of fit (skill vs seniority match) + noise
fit = -np.abs(df.skill_score * 5 - df.job_seniority) + df.exp_years * 0.2
fit += rng.normal(0, 2.0, len(df))

# From day 20 onward we inject feature drift: experience field starts being
# under-logged (a realistic serving bug) -> used later by drift_monitor.py
drift_mask = df.day >= 20
df.loc[drift_mask, "exp_years"] = (df.loc[drift_mask, "exp_years"] * 0.4).round()

# relevance label 0..3 (click, application, shortlist, hire-track) from fit
q = pd.qcut(fit.rank(method="first"), [0, .55, .8, .93, 1.0], labels=[0, 1, 2, 3])
df["relevance"] = q.astype(int)
df["shortlisted"] = 0  # placeholder, computed properly below
df["clicked"] = (df.relevance >= 1).astype(int)
df["applied"] = (df.relevance >= 2).astype(int)
df["shortlisted"] = (df.relevance >= 3).astype(int)

# a small, deliberate fairness gap in the LABELS (mirrors a real historic
# bias in shortlisting) so the fairness audit has something real to catch
bias_flip = (df.group == "B") & (rng.random(len(df)) < 0.06)
df.loc[bias_flip & (df.relevance == 3), "relevance"] = 2
df["shortlisted"] = (df.relevance >= 3).astype(int)

df["query_id"] = df["candidate_id"]  # ranking is per-candidate (jobs ranked for them)

candidates.to_csv(f"{OUT}/candidates.csv", index=False)
jobs.to_csv(f"{OUT}/jobs.csv", index=False)
df.to_csv(f"{OUT}/logs.csv", index=False)

print(f"candidates={len(candidates)} jobs={len(jobs)} impressions={len(df)}")
print(df.relevance.value_counts().to_dict())
