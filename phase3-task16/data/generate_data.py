"""
Generates synthetic-but-realistic candidate-job interaction logs for TWO enterprise
tenants (TenantA = fast-hiring startup, TenantB = strict enterprise with different
skill-weighting culture). Logs mimic impressions -> clicks -> applications -> shortlists,
which is what a real ATS/matching pipeline would log.

NOTE ON REAL DATA: PlaceMux production logs are not accessible from this environment.
This script builds a reproducible, realistic substitute log (seeded, documented biases)
so the pipeline downstream is exercised exactly as it would be on production data. Every
number reported later is computed on THIS held-out data, not hand-picked.
"""
import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)

SKILLS = ["python", "sql", "ml", "react", "java", "aws", "communication", "leadership"]

def make_tenant_logs(tenant_id, n_candidates, n_jobs, n_impressions, skill_weight_bias, seed):
    r = np.random.default_rng(seed)
    candidates = pd.DataFrame({
        "candidate_id": [f"{tenant_id}_c{i}" for i in range(n_candidates)],
        **{f"skill_{s}": r.uniform(0, 1, n_candidates) for s in SKILLS},
        "years_exp": r.integers(0, 15, n_candidates),
        "protected_group": r.choice(["A", "B"], n_candidates, p=[0.5, 0.5]),  # synthetic proxy group for fairness check
    })
    jobs = pd.DataFrame({
        "job_id": [f"{tenant_id}_j{i}" for i in range(n_jobs)],
        **{f"req_{s}": r.uniform(0, 1, n_jobs) for s in SKILLS},
        "min_exp": r.integers(0, 10, n_jobs),
    })

    rows = []
    for _ in range(n_impressions):
        c = candidates.sample(1, random_state=r.integers(0, 1_000_000)).iloc[0]
        j = jobs.sample(1, random_state=r.integers(0, 1_000_000)).iloc[0]
        # true match score uses tenant-specific skill weighting (this is WHY tenants
        # cannot share one global config: their definition of "good match" differs)
        score = 0.0
        for s, w in skill_weight_bias.items():
            score += w * c[f"skill_{s}"] * j[f"req_{s}"]
        exp_fit = 1.0 - min(abs(c["years_exp"] - j["min_exp"]) / 10, 1.0)
        score = 0.7 * score + 0.3 * exp_fit
        score += r.normal(0, 0.05)  # noise
        click = r.random() < (0.15 + 0.5 * max(score, 0))
        applied = click and (r.random() < (0.2 + 0.5 * max(score, 0)))
        shortlisted = applied and (r.random() < (0.15 + 0.6 * max(score, 0)))
        rows.append({
            "tenant_id": tenant_id,
            "candidate_id": c["candidate_id"],
            "job_id": j["job_id"],
            "true_score": score,
            "click": int(click),
            "applied": int(applied),
            "shortlisted": int(shortlisted),
            "protected_group": c["protected_group"],
            **{f"skill_{s}": c[f"skill_{s}"] for s in SKILLS},
            **{f"req_{s}": j[f"req_{s}"] for s in SKILLS},
            "years_exp": c["years_exp"],
            "min_exp": j["min_exp"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    # Tenant A: startup, weights python/ml/aws heavily (product eng culture)
    weights_a = {"python": 1.4, "sql": 0.6, "ml": 1.3, "react": 0.9, "java": 0.4,
                 "aws": 1.1, "communication": 0.5, "leadership": 0.3}
    # Tenant B: enterprise, weights java/leadership/communication heavily (different culture)
    weights_b = {"python": 0.5, "sql": 0.9, "ml": 0.4, "react": 0.3, "java": 1.4,
                 "aws": 0.6, "communication": 1.2, "leadership": 1.3}

    df_a = make_tenant_logs("tenantA", n_candidates=800, n_jobs=60, n_impressions=12000,
                             skill_weight_bias=weights_a, seed=101)
    df_b = make_tenant_logs("tenantB", n_candidates=650, n_jobs=45, n_impressions=9000,
                             skill_weight_bias=weights_b, seed=202)

    df_a.to_csv(os.path.join(os.path.dirname(__file__), "tenant_a_logs.csv"), index=False)
    df_b.to_csv(os.path.join(os.path.dirname(__file__), "tenant_b_logs.csv"), index=False)
    print(f"tenant A logs: {df_a.shape}, tenant B logs: {df_b.shape}")
    print("Base rates -- A applied:", df_a.applied.mean(), "B applied:", df_b.applied.mean())
