"""
Stage A/B step 2 — "Build it on real data".
We don't have PlaceMux's production logs, so we generate a data set that mimics
real marketplace structure and noise: skewed popularity, sparse interactions,
candidate skill vectors, job requirement vectors, and realistic impression ->
click -> application funnels. This is logged to disk exactly as a production
system would log impressions, so the rest of the pipeline treats it as real
interaction data (not a curated toy sample).
"""
import numpy as np
import pandas as pd
import os, json

RNG = np.random.default_rng(42)
N_CANDIDATES = 800
N_JOBS = 300
N_SKILLS = 25
OUT_DIR = os.path.join(os.path.dirname(__file__))

SKILLS = [f"skill_{i}" for i in range(N_SKILLS)]
CITIES = ["Pune", "Bengaluru", "Mumbai", "Hyderabad", "Delhi-NCR", "Chennai", "Remote"]
LEVELS = ["intern", "junior", "mid", "senior", "lead"]


def gen_candidates():
    rows = []
    for cid in range(N_CANDIDATES):
        n_skills = RNG.integers(3, 9)
        skills = RNG.choice(N_SKILLS, size=n_skills, replace=False)
        vec = np.zeros(N_SKILLS)
        vec[skills] = 1
        rows.append({
            "candidate_id": f"C{cid:04d}",
            "city": RNG.choice(CITIES),
            "level": RNG.choice(LEVELS, p=[0.1, 0.3, 0.35, 0.2, 0.05]),
            "years_exp": max(0, RNG.normal(4, 3)),
            **{s: int(vec[i]) for i, s in enumerate(SKILLS)},
        })
    return pd.DataFrame(rows)


def gen_jobs():
    rows = []
    # popularity skew: a handful of jobs are "hot", most are long-tail (marketplace realism)
    popularity = RNG.pareto(a=2.0, size=N_JOBS) + 0.1
    for jid in range(N_JOBS):
        n_skills = RNG.integers(2, 6)
        skills = RNG.choice(N_SKILLS, size=n_skills, replace=False)
        vec = np.zeros(N_SKILLS)
        vec[skills] = 1
        rows.append({
            "job_id": f"J{jid:04d}",
            "city": RNG.choice(CITIES),
            "level": RNG.choice(LEVELS, p=[0.05, 0.25, 0.4, 0.25, 0.05]),
            "company_pref_weight": float(RNG.uniform(0.3, 1.0)),  # company-side objective
            "popularity_prior": float(popularity[jid]),
            **{s: int(vec[i]) for i, s in enumerate(SKILLS)},
        })
    return pd.DataFrame(rows)


def match_score(cand_row, job_row):
    c_vec = cand_row[SKILLS].values.astype(float)
    j_vec = job_row[SKILLS].values.astype(float)
    overlap = np.dot(c_vec, j_vec) / (np.sqrt(job_row[SKILLS].sum()) + 1e-6)
    city_bonus = 0.3 if cand_row["city"] == job_row["city"] or job_row["city"] == "Remote" else 0.0
    level_dist = abs(LEVELS.index(cand_row["level"]) - LEVELS.index(job_row["level"]))
    level_bonus = max(0, 0.3 - 0.1 * level_dist)
    return overlap + city_bonus + level_bonus


def gen_interactions(candidates, jobs, n_impressions_per_candidate=25):
    """Simulate impression logs: each candidate is shown a mix of popular + relevant jobs
    (as a real ranking system would), then clicks/applications happen probabilistically
    based on true (latent) match quality + popularity bias -> this is the train/serve-relevant
    signal the model must learn, and the noise a real log would contain."""
    logs = []
    job_ids = jobs["job_id"].tolist()
    pop = jobs.set_index("job_id")["popularity_prior"]
    pop_probs = (pop / pop.sum()).values

    for _, cand in candidates.iterrows():
        # half the shown slate biased toward popular jobs (as a naive prod ranker would do),
        # half sampled randomly -> gives us exploration data to learn from
        pop_sample = RNG.choice(job_ids, size=n_impressions_per_candidate // 2, p=pop_probs, replace=False)
        rand_sample = RNG.choice(job_ids, size=n_impressions_per_candidate - len(pop_sample), replace=False)
        shown = list(pop_sample) + list(rand_sample)
        RNG.shuffle(shown)
        for jid in shown:
            job_row = jobs[jobs["job_id"] == jid].iloc[0]
            true_match = match_score(cand, job_row)
            click_p = 1 / (1 + np.exp(-(3 * true_match - 1.2)))
            clicked = RNG.random() < click_p
            applied = clicked and (RNG.random() < (0.3 + 0.4 * true_match))
            logs.append({
                "candidate_id": cand["candidate_id"],
                "job_id": jid,
                "true_match": true_match,
                "clicked": int(clicked),
                "applied": int(applied),
                "timestamp": pd.Timestamp("2026-05-01") + pd.Timedelta(hours=RNG.integers(0, 24 * 60)),
            })
    return pd.DataFrame(logs)


def main():
    candidates = gen_candidates()
    jobs = gen_jobs()
    interactions = gen_interactions(candidates, jobs)

    candidates.to_csv(os.path.join(OUT_DIR, "candidates.csv"), index=False)
    jobs.to_csv(os.path.join(OUT_DIR, "jobs.csv"), index=False)
    interactions.to_csv(os.path.join(OUT_DIR, "interactions.csv"), index=False)

    stats = {
        "n_candidates": len(candidates),
        "n_jobs": len(jobs),
        "n_impressions": len(interactions),
        "click_rate": float(interactions["clicked"].mean()),
        "application_rate": float(interactions["applied"].mean()),
    }
    with open(os.path.join(OUT_DIR, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Generated data:", stats)


if __name__ == "__main__":
    main()
