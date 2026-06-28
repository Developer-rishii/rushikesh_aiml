"""
PlaceMux Quality Sign-Off - Synthetic Dataset Generator
=========================================================
Generates real-shaped sample data for regression sign-off testing of the
matching/recommendation system after monetization integration.

Outputs (CSV):
  data/students.csv        - 200 student profiles
  data/jobs.csv            - 75 job descriptions
  data/monetization_events.csv - 500+ payment events with deliberate edge cases
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Skill universe
# ---------------------------------------------------------------------------
SKILLS = [
    "Python", "JavaScript", "React", "Node.js", "SQL",
    "Docker", "AWS", "Machine Learning",
]
NUM_SKILLS = len(SKILLS)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def generate_students(n: int = 200) -> pd.DataFrame:
    """Generate n student profiles with verified skill scores (0-5) and years of exposure."""
    rows = []
    for i in range(n):
        sid = f"S{i+1:03d}"
        scores = {}
        # Each student has 4-8 skills; some deliberately left as NaN
        num_skills_known = random.randint(4, NUM_SKILLS)
        known_skills = random.sample(SKILLS, num_skills_known)
        for sk in SKILLS:
            if sk in known_skills:
                # ~10 % chance of a missing score even for a "known" skill (edge case)
                if random.random() < 0.10:
                    scores[sk] = np.nan
                else:
                    scores[sk] = random.randint(1, 5)
            else:
                scores[sk] = np.nan
        years = round(random.uniform(0.5, 12.0), 1)
        row = {"student_id": sid, "years_of_exposure": years}
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_jobs(n: int = 75) -> pd.DataFrame:
    """Generate n job descriptions with required skills, minimum levels, and price tier."""
    tiers = ["free", "basic", "premium"]
    rows = []
    for i in range(n):
        jid = f"J{i+1:03d}"
        num_req = random.randint(2, 6)
        req_skills = random.sample(SKILLS, min(num_req, NUM_SKILLS))

        # Edge case: one JD with zero overlapping required skills (uses fake skills)
        if i == 0:
            req_skills = ["Cobol", "Fortran", "VHDL"]

        req_levels = {sk: random.randint(2, 5) for sk in req_skills}
        tier = random.choice(tiers)
        rows.append({
            "job_id": jid,
            "required_skills": "|".join(req_skills),
            "required_levels": "|".join(f"{sk}:{lv}" for sk, lv in req_levels.items()),
            "price_tier": tier,
        })
    return pd.DataFrame(rows)


def generate_monetization_events(
    students: pd.DataFrame,
    jobs: pd.DataFrame,
    n: int = 550,
) -> pd.DataFrame:
    """Generate monetization event rows with deliberate edge cases."""
    statuses = ["success", "failed", "pending", "refunded"]
    status_weights = [0.70, 0.10, 0.08, 0.05]  # ~7 % will be duplicates added below

    student_ids = students["student_id"].tolist()
    job_ids = jobs["job_id"].tolist()
    tier_prices = {"free": 0.0, "basic": 29.99, "premium": 99.99}
    job_tier_map = dict(zip(jobs["job_id"], jobs["price_tier"]))

    rows = []
    for i in range(n):
        aid = f"A{i+1:04d}"
        sid = random.choice(student_ids)
        jid = random.choice(job_ids)
        status = random.choices(statuses, weights=status_weights, k=1)[0]
        tier = job_tier_map[jid]
        gateway_amt = tier_prices[tier]

        # ~5 % gateway/recorded amount mismatch (edge case)
        if random.random() < 0.05 and gateway_amt > 0:
            recorded_amt = round(gateway_amt + random.choice([-5.0, 5.0, 10.0, -10.0]), 2)
        else:
            recorded_amt = gateway_amt

        rows.append({
            "application_id": aid,
            "student_id": sid,
            "job_id": jid,
            "payment_status": status,
            "gateway_amount": gateway_amt,
            "recorded_amount": recorded_amt,
        })

    # Inject ~25 duplicate/partial payment events (edge case)
    for _ in range(25):
        base = random.choice(rows)
        dup = base.copy()
        dup["application_id"] = f"A{len(rows)+1:04d}"
        # Partial: sometimes only gateway recorded, amount zeroed
        if random.random() < 0.4:
            dup["recorded_amount"] = 0.0
            dup["payment_status"] = "pending"
        rows.append(dup)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    students = generate_students(200)
    jobs = generate_jobs(75)
    events = generate_monetization_events(students, jobs, 550)

    students.to_csv(os.path.join(OUTPUT_DIR, "students.csv"), index=False)
    jobs.to_csv(os.path.join(OUTPUT_DIR, "jobs.csv"), index=False)
    events.to_csv(os.path.join(OUTPUT_DIR, "monetization_events.csv"), index=False)

    print(f"[OK] Generated {len(students)} students  -> data/students.csv")
    print(f"[OK] Generated {len(jobs)} jobs          -> data/jobs.csv")
    print(f"[OK] Generated {len(events)} events      -> data/monetization_events.csv")

    # Quick sanity stats
    missing_pct = students[SKILLS].isna().mean().mean() * 100
    mismatches = (events["gateway_amount"] != events["recorded_amount"]).sum()
    failed = (events["payment_status"].isin(["failed", "pending"])).sum()
    print(f"  |  Missing skill scores: {missing_pct:.1f}%")
    print(f"  |  Amount mismatches:    {mismatches}")
    print(f"  `  Failed/pending pays:  {failed}")


if __name__ == "__main__":
    main()
