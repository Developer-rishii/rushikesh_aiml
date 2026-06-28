"""
PlaceMux Quality Sign-Off - Baseline Matcher
==============================================
Dumb baseline: rank candidates by raw overlap count of
(required skills & student's verified skills).  Binary prediction:
overlap_ratio >= 0.5 -> match = 1.

This is a fixed rule, NOT a trained model.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SKILLS = ["Python", "JavaScript", "React", "Node.js", "SQL", "Docker", "AWS", "Machine Learning"]


def parse_required_skills(row: pd.Series) -> list[str]:
    """Extract list of required skill names from a job row."""
    raw = row.get("required_skills", "")
    if pd.isna(raw) or raw == "":
        return []
    return [s.strip() for s in str(raw).split("|") if s.strip()]


def compute_baseline_score(student: pd.Series, job: pd.Series) -> dict:
    """
    Returns:
        overlap_count: int - number of required skills the student has (non-NaN score)
        total_required: int
        overlap_ratio: float
        is_match: int (1 if overlap_ratio >= 0.5 else 0)
    """
    req_skills = parse_required_skills(job)
    total_required = len(req_skills)
    if total_required == 0:
        return {"overlap_count": 0, "total_required": 0,
                "overlap_ratio": 0.0, "is_match": 0}

    overlap = 0
    for sk in req_skills:
        if sk in student.index and pd.notna(student[sk]) and student[sk] > 0:
            overlap += 1

    ratio = overlap / total_required
    return {
        "overlap_count": overlap,
        "total_required": total_required,
        "overlap_ratio": round(ratio, 4),
        "is_match": int(ratio >= 0.5),
    }


def run_baseline(students: pd.DataFrame, jobs: pd.DataFrame,
                 events: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline predictions for every (student, job) pair in events."""
    student_map = students.set_index("student_id")
    job_map = jobs.set_index("job_id")

    results = []
    for _, ev in events.iterrows():
        sid, jid = ev["student_id"], ev["job_id"]
        if sid not in student_map.index or jid not in job_map.index:
            continue
        score = compute_baseline_score(student_map.loc[sid], job_map.loc[jid])
        score["student_id"] = sid
        score["job_id"] = jid
        score["application_id"] = ev["application_id"]
        results.append(score)

    return pd.DataFrame(results)


def main():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "monetization_events.csv"))

    preds = run_baseline(students, jobs, events)
    out_path = os.path.join(DATA_DIR, "baseline_predictions.csv")
    preds.to_csv(out_path, index=False)
    print(f"[OK] Baseline predictions: {len(preds)} rows -> data/baseline_predictions.csv")
    print(f"  |  Match rate: {preds['is_match'].mean():.2%}")
    print(f"  `  Mean overlap ratio: {preds['overlap_ratio'].mean():.3f}")


if __name__ == "__main__":
    main()
