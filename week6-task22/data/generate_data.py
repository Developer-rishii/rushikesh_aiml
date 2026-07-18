"""
data/generate_data.py

Generates real-shaped (synthetic) PlaceMux data:
  - students.csv        : verified skill scores per student
  - jobs.csv             : JD required skills + weights per job
  - interactions_monthN.csv : student-job pairs with a ground-truth
                              "good_match" label, for 6 rolling months.

Drift is INTENTIONALLY baked in from month 4 onward:
  - The market's skill demand shifts (a new skill "genai" rises fast,
    "jquery" fades).
  - The student population's skill distribution shifts to follow the
    market a step behind (realistic lag).
  - Label noise increases slightly (more borderline hires).

This gives the drift monitor something genuine to detect, and the
retraining pipeline a genuine reason to fire, instead of a toy demo.
"""
import numpy as np
import pandas as pd
import json
import os

RNG = np.random.default_rng(42)

SKILLS_CORE = [
    "python", "sql", "java", "react", "aws", "docker", "kubernetes",
    "nlp", "ml_fundamentals", "data_structures", "system_design",
    "spark", "excel", "communication", "flask", "jquery", "node",
    "cloud_security", "testing", "git",
]
SKILL_RISING = "genai"        # demand rises sharply from month 4
SKILL_FADING = "jquery"       # demand fades from month 4

N_STUDENTS = 1200
N_JOBS = 260
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
PAIRS_PER_MONTH = 1500

OUT_DIR = os.path.dirname(__file__)


def gen_students(n, month_idx):
    """Student skill vectors. From month 4, more students pick up genai
    (but with a realistic lag vs job market demand) and fewer pick up jquery."""
    rows = []
    genai_uptake = 0.05 if month_idx < 3 else min(0.55, 0.15 + 0.15 * (month_idx - 2))
    jquery_uptake = 0.35 if month_idx < 3 else max(0.08, 0.35 - 0.08 * (month_idx - 2))
    for i in range(n):
        n_skills = RNG.integers(5, 14)
        pool = list(SKILLS_CORE)
        chosen = list(RNG.choice(pool, size=min(n_skills, len(pool)), replace=False))
        if RNG.random() < genai_uptake:
            chosen.append(SKILL_RISING)
        if RNG.random() < jquery_uptake and SKILL_FADING not in chosen:
            chosen.append(SKILL_FADING)
        scores = {s: round(float(np.clip(RNG.normal(0.72, 0.15), 0.1, 1.0)), 3) for s in chosen}
        rows.append({
            "student_id": f"S{month_idx}_{i:05d}",
            "years_experience": int(np.clip(RNG.normal(3, 2), 0, 12)),
            "skills_json": json.dumps(scores),
        })
    return pd.DataFrame(rows)


def gen_jobs(n, month_idx):
    """JDs. From month 4, genai becomes a common requirement, jquery rarer."""
    rows = []
    genai_demand = 0.03 if month_idx < 3 else min(0.6, 0.2 + 0.18 * (month_idx - 2))
    jquery_demand = 0.3 if month_idx < 3 else max(0.05, 0.3 - 0.07 * (month_idx - 2))
    for i in range(n):
        n_req = RNG.integers(3, 7)
        pool = list(SKILLS_CORE)
        chosen = list(RNG.choice(pool, size=min(n_req, len(pool)), replace=False))
        if RNG.random() < genai_demand:
            chosen.append(SKILL_RISING)
        if RNG.random() < jquery_demand and SKILL_FADING not in chosen:
            chosen.append(SKILL_FADING)
        weights = {s: round(float(RNG.uniform(0.4, 1.0)), 2) for s in chosen}
        rows.append({
            "job_id": f"J{month_idx}_{i:05d}",
            "years_required": int(np.clip(RNG.normal(2.5, 1.8), 0, 8)),
            "required_skills_json": json.dumps(weights),
        })
    return pd.DataFrame(rows)


def true_match_probability(student_skills, job_skills, years_gap, month_idx):
    """Latent ground-truth function the labels are drawn from.
    Not visible to the model -- this is what 'reality' looks like."""
    if not job_skills:
        return 0.02
    overlap = set(student_skills) & set(job_skills)
    weight_sum = sum(job_skills.values())
    weighted = sum(student_skills[s] * job_skills[s] for s in overlap) / weight_sum if weight_sum else 0
    exp_penalty = 0.0 if years_gap >= 0 else min(0.4, -years_gap * 0.12)
    base = 0.85 * weighted - exp_penalty
    # label noise creeps up slightly in later months (real hiring is messier)
    noise_scale = 0.05 if month_idx < 3 else 0.05 + 0.01 * (month_idx - 2)
    noisy = base + RNG.normal(0, noise_scale)
    return float(np.clip(noisy, 0, 1))


def gen_interactions(students_df, jobs_df, n_pairs, month_idx):
    rows = []
    sids = students_df["student_id"].values
    jids = jobs_df["job_id"].values
    s_lookup = students_df.set_index("student_id")
    j_lookup = jobs_df.set_index("job_id")
    for _ in range(n_pairs):
        sid = RNG.choice(sids)
        jid = RNG.choice(jids)
        s_skills = json.loads(s_lookup.loc[sid, "skills_json"])
        j_skills = json.loads(j_lookup.loc[jid, "required_skills_json"])
        years_gap = int(s_lookup.loc[sid, "years_experience"]) - int(j_lookup.loc[jid, "years_required"])
        p = true_match_probability(s_skills, j_skills, years_gap, month_idx)
        label = int(RNG.random() < p)
        rows.append({"student_id": sid, "job_id": jid, "month": MONTHS[month_idx], "good_match": label})
    return pd.DataFrame(rows)


def main():
    all_students, all_jobs, all_interactions = [], [], []
    for m_idx, month in enumerate(MONTHS):
        students = gen_students(N_STUDENTS // len(MONTHS), m_idx)
        jobs = gen_jobs(N_JOBS // len(MONTHS), m_idx)
        interactions = gen_interactions(students, jobs, PAIRS_PER_MONTH, m_idx)
        all_students.append(students)
        all_jobs.append(jobs)
        all_interactions.append(interactions)
        interactions.to_csv(os.path.join(OUT_DIR, f"interactions_{month}.csv"), index=False)

    students_df = pd.concat(all_students, ignore_index=True)
    jobs_df = pd.concat(all_jobs, ignore_index=True)
    students_df.to_csv(os.path.join(OUT_DIR, "students.csv"), index=False)
    jobs_df.to_csv(os.path.join(OUT_DIR, "jobs.csv"), index=False)

    print(f"Generated {len(students_df)} students, {len(jobs_df)} jobs, "
          f"{sum(len(x) for x in all_interactions)} interactions across {len(MONTHS)} months.")
    for m_idx, month in enumerate(MONTHS):
        pos_rate = all_interactions[m_idx]["good_match"].mean()
        print(f"  {month}: positive rate = {pos_rate:.3f}")


if __name__ == "__main__":
    main()
