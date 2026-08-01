"""
Generates realistic-shaped candidate-job interaction logs (impressions -> clicks ->
applications -> shortlists) for training/evaluating the matching model.

Not hand-curated per-row: we simulate an underlying "true affinity" function the
model has to recover from noisy behavioural signals, exactly like real logs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_CANDIDATES = 400
N_JOBS = 120
N_IMPRESSIONS = 20000

SKILLS = ["python", "sql", "react", "java", "aws", "ml", "node", "excel", "sales", "figma"]


def build_pool(n, seed_offset):
    rng = np.random.default_rng(42 + seed_offset)
    skill_vecs = rng.integers(0, 2, size=(n, len(SKILLS)))
    seniority = rng.integers(0, 4, size=n)          # 0=intern..3=senior
    location = rng.integers(0, 5, size=n)            # 5 city buckets
    return skill_vecs, seniority, location


def main():
    cand_skills, cand_sen, cand_loc = build_pool(N_CANDIDATES, 1)
    job_skills, job_sen, job_loc = build_pool(N_JOBS, 2)

    rows = []
    for _ in range(N_IMPRESSIONS):
        c = RNG.integers(0, N_CANDIDATES)
        j = RNG.integers(0, N_JOBS)

        skill_overlap = int((cand_skills[c] & job_skills[j]).sum())
        seniority_gap = abs(int(cand_sen[c]) - int(job_sen[j]))
        same_location = int(cand_loc[c] == job_loc[j])
        recency_days = int(RNG.integers(0, 60))          # how old the posting is
        candidate_activity = float(RNG.beta(2, 5))         # how active the candidate is lately

        # ground-truth affinity (hidden from the model, drives labels)
        affinity = (
            1.4 * skill_overlap
            - 0.9 * seniority_gap
            + 1.1 * same_location
            - 0.02 * recency_days
            + 1.5 * candidate_activity
            + RNG.normal(0, 1.0)
        )
        p_click = 1 / (1 + np.exp(-(affinity - 2.0)))
        clicked = RNG.random() < p_click
        applied = clicked and (RNG.random() < 1 / (1 + np.exp(-(affinity - 3.2))))
        shortlisted = applied and (RNG.random() < 1 / (1 + np.exp(-(affinity - 4.0))))

        # relevance label used for ranking eval: 0 none,1 click,2 apply,3 shortlist
        label = 0 + int(clicked) + int(applied) + int(shortlisted)

        rows.append(dict(
            candidate_id=c, job_id=j,
            skill_overlap=skill_overlap, seniority_gap=seniority_gap,
            same_location=same_location, recency_days=recency_days,
            candidate_activity=round(candidate_activity, 4),
            clicked=int(clicked), applied=int(applied), shortlisted=int(shortlisted),
            label=label,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "interactions.csv", index=False)
    print(f"wrote {len(df)} rows, label distribution:\n{df.label.value_counts()}")


if __name__ == "__main__":
    main()
