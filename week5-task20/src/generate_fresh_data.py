"""
Generate a fresh validation dataset at 3-5x the scale of the original Task 16
training data, using a DIFFERENT random seed so the Rec v1 model has never seen it.

Original Task 16 data: seed=42, 88 students, 4 colleges, ~1,078 matching rows.
Fresh data: seed=2026, 350+ students, 8 colleges, 3,500+ matching rows.
"""
import pandas as pd
import numpy as np
import os
import random
import hashlib

# Deterministic but different from Task 16's seed=42
FRESH_SEED = 2026


def generate_fresh_data(data_dir: str):
    """Generate fresh matching + outcomes data at production-like scale."""
    np.random.seed(FRESH_SEED)
    random.seed(FRESH_SEED)

    os.makedirs(data_dir, exist_ok=True)

    # ── Colleges ──────────────────────────────────────────────────────────
    colleges = {
        # original 4 (kept so the model sees familiar college distributions)
        "college_A": {"size": "large",  "n_students": 60},
        "college_B": {"size": "large",  "n_students": 55},
        "college_C": {"size": "medium", "n_students": 45},
        "college_D": {"size": "small",  "n_students": 8},
        # 4 new colleges the model has never seen
        "college_E": {"size": "large",  "n_students": 50},
        "college_F": {"size": "medium", "n_students": 40},
        "college_G": {"size": "medium", "n_students": 35},
        "college_H": {"size": "small",  "n_students": 5},
        # edge case: college with zero students
        "college_EMPTY": {"size": "small", "n_students": 0},
    }

    # ── Students ──────────────────────────────────────────────────────────
    students = []
    seniority_levels = ["junior", "mid", "senior"]
    for c_id, info in colleges.items():
        for i in range(info["n_students"]):
            students.append({
                "student_id": f"fresh_{c_id}_{i}",
                "college_id": c_id,
                "seniority": random.choice(seniority_levels),
                "college_size": info["size"],
            })

    # edge-case students
    students.append({
        "student_id": "fresh_college_A_alllow",
        "college_id": "college_A",
        "seniority": "junior",
        "college_size": "large",
    })

    print(f"  Fresh students: {len(students)}")

    # ── Jobs (100 jobs — 2x original) ─────────────────────────────────────
    jobs = [f"job_{i}" for i in range(100)]

    # ── Matching rows ─────────────────────────────────────────────────────
    matching_rows = []
    outcomes_rows = []

    college_score_params = {
        "college_A": (0.55, 0.97, 0.65, 0.98),
        "college_B": (0.35, 0.88, 0.45, 0.92),
        "college_C": (0.25, 0.72, 0.35, 0.80),
        "college_D": (0.40, 0.85, 0.50, 0.90),
        "college_E": (0.45, 0.90, 0.55, 0.93),
        "college_F": (0.30, 0.78, 0.40, 0.85),
        "college_G": (0.38, 0.82, 0.48, 0.88),
        "college_H": (0.42, 0.86, 0.52, 0.90),
        "college_EMPTY": (0.40, 0.80, 0.50, 0.85),
    }

    for student in students:
        s_id = student["student_id"]
        c_id = student["college_id"]

        # all-low-score edge case
        if s_id.endswith("_alllow"):
            num_jobs = 8
            ms_lo, ms_hi, ai_lo, ai_hi = 0.05, 0.28, 0.30, 0.50
        else:
            num_jobs = random.randint(10, 18)
            ms_lo, ms_hi, ai_lo, ai_hi = college_score_params.get(
                c_id, (0.40, 0.80, 0.50, 0.85)
            )

        candidate_jobs = random.sample(jobs, min(num_jobs, len(jobs)))

        for j_id in candidate_jobs:
            jd_seniority_level = random.choice([1, 2, 3, 4, 5])
            years_exposure_avg = round(np.random.uniform(1.0, 5.0), 1)
            match_score = round(np.random.uniform(ms_lo, ms_hi), 3)
            ai_trust_score = round(np.random.uniform(ai_lo, ai_hi), 3)
            skill_overlap_count = max(1, int(match_score * 8) + random.randint(-1, 1))
            skill_gap_count = max(0, 8 - skill_overlap_count + random.randint(0, 3))
            verified_skill_count = skill_overlap_count + random.randint(0, 3)

            matching_rows.append({
                "student_id": s_id,
                "college_id": c_id,
                "job_id": j_id,
                "match_score": match_score,
                "skill_overlap_count": skill_overlap_count,
                "skill_gap_count": skill_gap_count,
                "years_exposure_avg": years_exposure_avg,
                "jd_seniority_level": jd_seniority_level,
                "verified_skill_count": verified_skill_count,
                "ai_trust_score": ai_trust_score,
            })

    df_matching = pd.DataFrame(matching_rows)

    # ── Outcomes ──────────────────────────────────────────────────────────
    for _, row in df_matching.iterrows():
        if random.random() > 0.60:
            continue
        seniority_match_val = 1.0 if abs(
            row["jd_seniority_level"] - round(row["years_exposure_avg"])
        ) <= 1 else 0.0
        gap_penalty = row["skill_gap_count"] / 10.0
        composite = (
            row["match_score"] * 0.35
            + row["ai_trust_score"] * 0.25
            + seniority_match_val * 0.25
            - gap_penalty * 0.15
        )
        prob = np.clip(composite, 0.05, 0.95)
        outcome = int(np.random.binomial(1, prob))
        outcomes_rows.append({
            "student_id": row["student_id"],
            "job_id": row["job_id"],
            "outcome": outcome,
        })

    df_outcomes = pd.DataFrame(outcomes_rows)

    # ── Student metadata (for segmented analysis) ─────────────────────────
    students_meta = pd.DataFrame(students)

    # ── Persist ───────────────────────────────────────────────────────────
    df_matching.to_csv(os.path.join(data_dir, "fresh_matching.csv"), index=False)
    df_outcomes.to_csv(os.path.join(data_dir, "fresh_outcomes.csv"), index=False)
    students_meta.to_csv(os.path.join(data_dir, "fresh_students_meta.csv"), index=False)

    # Hash for verification that this is genuinely different data
    data_hash = hashlib.sha256(
        df_matching.to_csv(index=False).encode()
    ).hexdigest()[:16]

    print(f"  Fresh matching rows : {len(df_matching)}")
    print(f"  Fresh outcome rows  : {len(df_outcomes)}")
    print(f"  Colleges            : {df_matching['college_id'].nunique()}")
    print(f"  Students            : {df_matching['student_id'].nunique()}")
    print(f"  Data hash           : {data_hash}")

    return {
        "matching_rows": len(df_matching),
        "outcome_rows": len(df_outcomes),
        "colleges": int(df_matching["college_id"].nunique()),
        "students": int(df_matching["student_id"].nunique()),
        "data_hash": data_hash,
        "seed": FRESH_SEED,
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_fresh_data(os.path.join(base, "data"))
