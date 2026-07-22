import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
SKILLS = ["python", "sql", "react", "node", "aws", "ml", "java", "spring",
          "figma", "sales", "excel", "communication", "devops", "k8s", "go"]

def make_jobs(n_jobs=500):
    rows = []
    for jid in range(n_jobs):
        n_skills = RNG.integers(2, 6)
        job_skills = RNG.choice(SKILLS, size=n_skills, replace=False)
        popularity = RNG.pareto(a=1.5) + 1  # power-law popularity prior
        rows.append({"job_id": jid, "skills": list(job_skills), "popularity": popularity})
    df = pd.DataFrame(rows)
    df["popularity"] = df["popularity"] / df["popularity"].sum()
    return df

def make_users(n_users=2000, cold_start_frac=0.3):
    rows = []
    for uid in range(n_users):
        n_skills = RNG.integers(1, 5)
        user_skills = RNG.choice(SKILLS, size=n_skills, replace=False)
        is_cold_start = RNG.random() < cold_start_frac
        rows.append({"user_id": uid, "skills": list(user_skills), "is_cold_start": is_cold_start})
    return pd.DataFrame(rows)

def skill_overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def make_interactions(users, jobs, avg_impressions_warm=25, avg_impressions_cold=0):
    """Cold-start users get 0 PRIOR interactions (that's the definition) but we still
    generate their FIRST SESSION separately in evaluate.py so we can measure lift on it."""
    rows = []
    for _, u in users.iterrows():
        if u["is_cold_start"]:
            continue  # no history, by definition
        n_impr = RNG.poisson(avg_impressions_warm)
        shown = jobs.sample(n=min(n_impr, len(jobs)), weights=jobs["popularity"], random_state=None)
        for _, j in shown.iterrows():
            overlap = skill_overlap(u["skills"], j["skills"])
            click_p = np.clip(0.05 + 0.6 * overlap + 0.1 * j["popularity"] * 50, 0, 0.9)
            clicked = RNG.random() < click_p
            applied = clicked and (RNG.random() < 0.3 + 0.4 * overlap)
            rows.append({"user_id": u["user_id"], "job_id": j["job_id"],
                         "clicked": clicked, "applied": applied, "overlap": overlap})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = Path(__file__).parent
    jobs = make_jobs()
    users = make_users()
    interactions = make_interactions(users, jobs)
    jobs.to_json(out / "jobs.json", orient="records")
    users.to_json(out / "users.json", orient="records")
    interactions.to_csv(out / "interactions.csv", index=False)
    print(f"jobs={len(jobs)} users={len(users)} "
          f"cold_start_users={users['is_cold_start'].sum()} "
          f"logged_interactions={len(interactions)}")