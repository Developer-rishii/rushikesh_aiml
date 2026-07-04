import pandas as pd
import numpy as np
import os
import random

def generate_data():
    """Generate realistic multi-college matching data with deliberate edge cases."""
    np.random.seed(42)
    random.seed(42)
    
    # ---- Student roster ----
    students = []
    # College A: 25 strong students (high avg match_score)
    for i in range(25):
        students.append({"student_id": f"student_A_{i}", "college_id": "college_A"})
    # College B: 30 average students
    for i in range(30):
        students.append({"student_id": f"student_B_{i}", "college_id": "college_B"})
    # College C: 30 students with more skill gaps
    for i in range(30):
        students.append({"student_id": f"student_C_{i}", "college_id": "college_C"})
    # College D: 2 students (small-college edge case)
    students.append({"student_id": "student_D_0", "college_id": "college_D"})
    students.append({"student_id": "student_D_1", "college_id": "college_D"})
    # Edge case: all-low-score student
    students.append({"student_id": "student_A_low", "college_id": "college_A"})
    # Edge case: zero-candidate student (will have NO rows in matching_v1)
    students.append({"student_id": "student_B_zero", "college_id": "college_B"})

    # ---- Job roster (50 jobs, available to all colleges) ----
    job_titles = {
        "job_0": "Data Analyst at Infosys",
        "job_1": "ML Engineer at Wipro",
        "job_2": "Backend Dev at TCS",
        "job_3": "Full-Stack Dev at Cognizant",
        "job_4": "DevOps Engineer at HCL",
        "job_5": "Cloud Architect at Mindtree",
        "job_6": "QA Engineer at Mphasis",
        "job_7": "Product Analyst at Flipkart",
        "job_8": "Data Scientist at Fractal",
        "job_9": "Software Engineer at Google",
    }
    for i in range(10, 50):
        job_titles[f"job_{i}"] = f"Role_{i} at Company_{i}"
    jobs = list(job_titles.keys())

    matching_rows = []
    outcomes_rows = []

    for student in students:
        s_id = student["student_id"]
        c_id = student["college_id"]

        # Zero-candidate edge case: skip entirely
        if s_id == "student_B_zero":
            continue

        # Number of candidate jobs per student (10-15 so top-5 is selective)
        num_jobs = random.randint(10, 15)
        # For the all-low student, give them 8 jobs
        if s_id == "student_A_low":
            num_jobs = 8
        candidate_jobs = random.sample(jobs, num_jobs)

        for j_idx, j_id in enumerate(candidate_jobs):
            jd_seniority_level = random.choice([1, 2, 3, 4, 5])
            years_exposure_avg = round(np.random.uniform(1.0, 5.0), 1)

            # College-specific score distributions
            if s_id == "student_A_low":
                match_score = round(np.random.uniform(0.05, 0.28), 3)
                ai_trust_score = round(np.random.uniform(0.3, 0.5), 3)
            elif c_id == "college_A":
                match_score = round(np.random.uniform(0.55, 0.97), 3)
                ai_trust_score = round(np.random.uniform(0.65, 0.98), 3)
            elif c_id == "college_C":
                match_score = round(np.random.uniform(0.25, 0.72), 3)
                ai_trust_score = round(np.random.uniform(0.35, 0.80), 3)
            elif c_id == "college_D":
                match_score = round(np.random.uniform(0.40, 0.85), 3)
                ai_trust_score = round(np.random.uniform(0.50, 0.90), 3)
            else:  # college_B
                match_score = round(np.random.uniform(0.35, 0.88), 3)
                ai_trust_score = round(np.random.uniform(0.45, 0.92), 3)

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

    # ---- Placement outcomes (training signal) ----
    # Generate outcomes for ~60% of student-job pairs.
    for _, row in df_matching.iterrows():
        if random.random() > 0.60:
            continue
        # Composite probability of a good outcome
        seniority_match = 1.0 if abs(row["jd_seniority_level"] - round(row["years_exposure_avg"])) <= 1 else 0.0
        gap_penalty = row["skill_gap_count"] / 10.0
        composite = (
            row["match_score"] * 0.35
            + row["ai_trust_score"] * 0.25
            + seniority_match * 0.25
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

    os.makedirs("data", exist_ok=True)
    df_matching.to_csv("data/matching_v1_output.csv", index=False)
    df_outcomes.to_csv("data/placement_outcomes.csv", index=False)

    print(f"Generated matching_v1_output.csv: {len(df_matching)} rows, "
          f"{df_matching['student_id'].nunique()} students, "
          f"{df_matching['college_id'].nunique()} colleges")
    print(f"Generated placement_outcomes.csv: {len(df_outcomes)} rows "
          f"({len(df_outcomes[df_outcomes['outcome']==1])} positive, "
          f"{len(df_outcomes[df_outcomes['outcome']==0])} negative)")
    print(f"Edge cases present:")
    print(f"  - student_A_low (all low scores): "
          f"{len(df_matching[df_matching['student_id']=='student_A_low'])} candidates")
    print(f"  - student_B_zero (zero candidates): 0 candidates (not in file)")
    print(f"  - college_D (small college): "
          f"{df_matching[df_matching['college_id']=='college_D']['student_id'].nunique()} students")

if __name__ == "__main__":
    generate_data()
