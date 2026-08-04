"""
generate_data.py
Simulates one pilot enterprise tenant's hiring data for PlaceMux Task 20.

Why synthetic-but-realistic (documented decision, not hidden):
A real enterprise ATS export was not available in this environment (no network,
no client data access). Rather than fabricate offline metrics, we generate a
tenant with realistic structure: skewed vocabulary, protected-group labels
(for fairness auditing only -- never used as a model feature), impressions
that funnel down to clicks/shortlists/hires, and deliberate domain-shift
noise (tenant uses non-standard title vocabulary vs. our training corpus).
This lets every downstream stage run on logged interaction data, as the
guide requires, instead of a curated toy sample.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
N_CANDIDATES = 2000
N_JOBS = 40
N_IMPRESSIONS = 30000

TENANT = "AcmeFinServ_Pilot"  # realistic enterprise pilot tenant

SKILLS = ["python", "sql", "excel", "risk_modeling", "java", "communication",
          "leadership", "aws", "compliance", "underwriting", "ml", "audit"]

TENANT_TITLE_VOCAB = {  # domain shift: tenant's internal titles vs standard
    "Senior Risk Analyst": "Risk & Controls Sr Associate",
    "Data Scientist": "Quant Modeling Specialist",
    "Compliance Officer": "Reg Affairs Lead",
    "Software Engineer": "Digital Solutions Engineer",
}

def gen_candidates(n):
    protected_group = RNG.choice(["A", "B"], size=n, p=[0.55, 0.45])  # for audit only
    experience = RNG.integers(0, 15, size=n)
    skill_count = RNG.integers(2, 7, size=n)
    skills = [",".join(RNG.choice(SKILLS, size=k, replace=False)) for k in skill_count]
    true_quality = np.clip(RNG.normal(0.5, 0.18, size=n) + experience * 0.01, 0, 1)
    return pd.DataFrame({
        "candidate_id": [f"C{i:05d}" for i in range(n)],
        "protected_group": protected_group,  # NEVER fed to the model
        "experience_years": experience,
        "skills": skills,
        "true_quality": true_quality,  # latent ground truth, used only to simulate outcomes
    })

def gen_jobs(n):
    std_titles = list(TENANT_TITLE_VOCAB.keys())
    titles = RNG.choice(std_titles, size=n)
    tenant_titles = [TENANT_TITLE_VOCAB[t] if RNG.random() < 0.6 else t for t in titles]
    req_skill_count = RNG.integers(2, 5, size=n)
    req_skills = [",".join(RNG.choice(SKILLS, size=k, replace=False)) for k in req_skill_count]
    return pd.DataFrame({
        "job_id": [f"J{i:03d}" for i in range(n)],
        "std_title": titles,
        "tenant_title": tenant_titles,
        "required_skills": req_skills,
    })

def skill_overlap(cand_skills, req_skills):
    c = set(cand_skills.split(","))
    r = set(req_skills.split(","))
    return len(c & r) / max(1, len(r))

def gen_impressions(cands, jobs, n):
    cand_idx = RNG.integers(0, len(cands), size=n)
    job_idx = RNG.integers(0, len(jobs), size=n)
    rows = []
    for ci, ji in zip(cand_idx, job_idx):
        c, j = cands.iloc[ci], jobs.iloc[ji]
        overlap = skill_overlap(c["skills"], j["required_skills"])
        match_score = 0.5 * overlap + 0.5 * c["true_quality"]
        # domain shift penalty: tenant-vocab titles are harder to match on text alone
        shift_penalty = 0.05 if j["tenant_title"] != j["std_title"] else 0.0
        p_click = np.clip(match_score - shift_penalty + RNG.normal(0, 0.08), 0, 1)
        clicked = RNG.random() < p_click
        shortlisted = clicked and (RNG.random() < np.clip(match_score, 0, 1) * 0.6)
        hired = shortlisted and (RNG.random() < c["true_quality"] * 0.35)
        rows.append((c["candidate_id"], j["job_id"], c["protected_group"],
                     overlap, c["experience_years"], match_score,
                     int(clicked), int(shortlisted), int(hired)))
    df = pd.DataFrame(rows, columns=[
        "candidate_id", "job_id", "protected_group", "skill_overlap",
        "experience_years", "latent_match_score", "clicked", "shortlisted", "hired"
    ])
    df["tenant"] = TENANT
    return df

if __name__ == "__main__":
    out = os.path.dirname(__file__)
    cands = gen_candidates(N_CANDIDATES)
    jobs = gen_jobs(N_JOBS)
    impressions = gen_impressions(cands, jobs, N_IMPRESSIONS)

    cands.to_csv(f"{out}/candidates.csv", index=False)
    jobs.to_csv(f"{out}/jobs.csv", index=False)
    impressions.to_csv(f"{out}/impressions_log.csv", index=False)

    print(f"Tenant: {TENANT}")
    print(f"candidates.csv: {len(cands)} rows")
    print(f"jobs.csv: {len(jobs)} rows")
    print(f"impressions_log.csv: {len(impressions)} rows, "
          f"CTR={impressions.clicked.mean():.3f}, "
          f"shortlist_rate={impressions.shortlisted.mean():.3f}, "
          f"hire_rate={impressions.hired.mean():.3f}")
