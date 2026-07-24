"""
Task 9 - Stage B/C data foundation.
Generates a realistic (synthetic, seeded) job-marketplace interaction dataset:
candidates x job impressions, with a *latent true relevance* that the model
never sees directly (mirrors reality: PlaceMux never observes ground-truth
'this is the best candidate for this job', only downstream behaviour).

Not curated/cherry-picked: we sample noisy behaviour from the latent
relevance, exactly like real impression/click/apply logs would look.
"""
import numpy as np
import pandas as pd
import hashlib

RNG_SEED = 42
N_USERS = 4000          # recruiters/candidates issuing searches
N_JOBS = 250
DAYS = 14
IMPRESSIONS_PER_USER_PER_DAY = (1, 5)  # uniform range


def _hash_bucket(key: str, salt: str, buckets: int = 10000) -> int:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h, 16) % buckets


def generate(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Candidate/job feature universe -----------------------------------
    users = pd.DataFrame({
        "user_id": [f"u{idx:05d}" for idx in range(N_USERS)],
        # synthetic, non-identifying cohort label used ONLY for the fairness
        # audit (stand-in for a protected attribute, deliberately abstracted
        # per the DPDP note in the study guide - we never store real PII).
        "segment": rng.choice(["A", "B"], size=N_USERS, p=[0.55, 0.45]),
        "experience_yrs": rng.integers(0, 15, size=N_USERS),
    })

    jobs = pd.DataFrame({
        "job_id": [f"j{idx:04d}" for idx in range(N_JOBS)],
        "seniority": rng.integers(0, 15, size=N_JOBS),
    })

    rows = []
    for day in range(DAYS):
        for _, u in users.iterrows():
            n_impr = rng.integers(*IMPRESSIONS_PER_USER_PER_DAY)
            if n_impr == 0:
                continue
            job_sample = jobs.sample(n=n_impr, random_state=int(rng.integers(0, 1_000_000)))
            for _, j in job_sample.iterrows():
                skill_match = float(np.clip(rng.normal(0.6, 0.2), 0, 1))
                exp_gap = -abs(u.experience_yrs - j.seniority) / 15.0
                location_match = float(rng.choice([1, 0], p=[0.7, 0.3]))
                recency = float(rng.uniform(0, 1))

                # LATENT true relevance PlaceMux is trying to predict.
                true_relevance = (
                    0.5 * skill_match
                    + 0.3 * (1 + exp_gap)
                    + 0.15 * location_match
                    + 0.05 * recency
                    + rng.normal(0, 0.05)
                )
                true_relevance = float(np.clip(true_relevance, 0, 1))

                rows.append({
                    "day": day,
                    "user_id": u.user_id,
                    "job_id": j.job_id,
                    "segment": u.segment,
                    "skill_match": round(skill_match, 4),
                    "exp_gap": round(exp_gap, 4),
                    "location_match": location_match,
                    "recency": round(recency, 4),
                    "true_relevance": round(true_relevance, 4),
                })

    df = pd.DataFrame(rows)
    df["assign_bucket"] = df["user_id"].apply(lambda u: _hash_bucket(u, "placemux-exp-v1"))
    return df


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = DATA_DIR / "interaction_logs.csv"
    
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"generated {len(df)} rows -> data/interaction_logs.csv")
    print(df.head())
