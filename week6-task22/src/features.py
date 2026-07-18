"""
src/features.py

Feature space definition for PlaceMux match scoring.
Deliberately kept small and *explainable* -- every feature has a plain
English meaning, because a black-box feature space defeats the point
of a hiring product (see Study Guide Section 4: Explainability).
"""
import json
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "overlap_count",
    "overlap_ratio",
    "weighted_skill_score",
    "years_gap",
    "missing_top_skill",
    "jd_breadth",
    "student_breadth",
]


class FeatureError(ValueError):
    """Raised when an interaction row can't be turned into features
    (bad input contract) -- see Pitfall: 'we can tighten security after
    launch'. We validate now, not later."""


def _safe_json_load(raw, field_name, row_id):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raise FeatureError(f"{field_name} missing for {row_id}")
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise FeatureError(f"{field_name} malformed for {row_id}: {e}")
    if not isinstance(obj, dict):
        raise FeatureError(f"{field_name} must be an object for {row_id}")
    return obj


def build_features_row(student_skills: dict, job_skills: dict, years_gap: float) -> dict:
    """Pure function: skills dicts + years gap -> feature dict.
    Handles the edge cases explicitly (empty skills, empty JD)."""
    if not job_skills:
        # Cold-start / malformed JD: no requirements at all.
        return dict(overlap_count=0, overlap_ratio=0.0, weighted_skill_score=0.0,
                    years_gap=years_gap, missing_top_skill=1, jd_breadth=0,
                    student_breadth=len(student_skills))

    overlap = set(student_skills) & set(job_skills)
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / len(job_skills)

    weight_sum = sum(job_skills.values())
    weighted_skill_score = (
        sum(student_skills[s] * job_skills[s] for s in overlap) / weight_sum
        if weight_sum > 0 else 0.0
    )

    top_skill = max(job_skills, key=job_skills.get)
    missing_top_skill = int(top_skill not in student_skills)

    return dict(
        overlap_count=overlap_count,
        overlap_ratio=overlap_ratio,
        weighted_skill_score=weighted_skill_score,
        years_gap=years_gap,
        missing_top_skill=missing_top_skill,
        jd_breadth=len(job_skills),
        student_breadth=len(student_skills),
    )


def build_feature_matrix(interactions: pd.DataFrame, students: pd.DataFrame,
                          jobs: pd.DataFrame, drop_bad_rows: bool = True) -> pd.DataFrame:
    """Joins interactions -> students/jobs -> feature matrix.
    Bad rows (malformed skills JSON, unknown ids) are logged and either
    dropped or raised, per drop_bad_rows -- this is the
    'errors handled, not silently ignored' requirement."""
    s_idx = students.set_index("student_id")
    j_idx = jobs.set_index("job_id")
    feats, errors = [], []

    for _, row in interactions.iterrows():
        rid = f"{row['student_id']}::{row['job_id']}"
        try:
            if row["student_id"] not in s_idx.index:
                raise FeatureError(f"unknown student_id {row['student_id']}")
            if row["job_id"] not in j_idx.index:
                raise FeatureError(f"unknown job_id {row['job_id']}")
            s_skills = _safe_json_load(s_idx.loc[row["student_id"], "skills_json"], "skills_json", rid)
            j_skills = _safe_json_load(j_idx.loc[row["job_id"], "required_skills_json"], "required_skills_json", rid)
            years_gap = (int(s_idx.loc[row["student_id"], "years_experience"])
                         - int(j_idx.loc[row["job_id"], "years_required"]))
            f = build_features_row(s_skills, j_skills, years_gap)
            f["student_id"] = row["student_id"]
            f["job_id"] = row["job_id"]
            if "month" in row:
                f["month"] = row["month"]
            if "good_match" in row:
                f["good_match"] = row["good_match"]
            feats.append(f)
        except FeatureError as e:
            errors.append(str(e))
            if not drop_bad_rows:
                raise

    if errors:
        print(f"[features] skipped {len(errors)} malformed row(s), e.g. {errors[:3]}")

    return pd.DataFrame(feats)
