"""
The ONE place feature logic is written. This module is imported by both
training/train_model.py and serving/model_service.py so that features can
never silently diverge between train-time and serve-time computation
(train/serve skew) — the study guide calls this out as "the single
biggest silent killer".

FEATURE_PIPELINE_VERSION is bumped whenever the logic changes, and is
logged alongside every prediction and every trained model, so a bad
score can always be traced back to the exact feature logic that produced it.
"""
import numpy as np
import pandas as pd

FEATURE_PIPELINE_VERSION = "fp_v1.0.0"

FEATURE_COLUMNS = [
    "skill_overlap_ratio",
    "exp_gap",
    "salary_gap_abs",
    "region_match",
    "job_popularity_log",
    "cand_activity_score",
]


def _skill_overlap_ratio(cand_skills_str: str, job_req_skills_str: str) -> float:
    cand = set(cand_skills_str.split("|")) if cand_skills_str else set()
    req = set(job_req_skills_str.split("|")) if job_req_skills_str else set()
    if not req:
        return 0.0
    return len(cand & req) / len(req)


def compute_features(row: dict) -> dict:
    """Compute the model-ready feature vector for a single (candidate, job)
    pair. `row` must contain the raw fields listed below, in either a
    training-log row (dict/Series) or a serving-time request payload —
    both call this exact function so behavior is identical.
    """
    skill_overlap = _skill_overlap_ratio(
        row["cand_skills"], row["job_req_skills"]
    )
    exp_gap = float(row["cand_experience_yrs"]) - float(row["job_min_exp"])
    salary_gap_abs = abs(float(row["cand_expected_salary"]) - float(row["job_salary_offered"]))
    region_match = 1.0 if row["cand_region"] == row["job_region"] else 0.0
    job_popularity_log = float(np.log1p(float(row["job_popularity"])))
    cand_activity_score = float(row["cand_activity_score"])

    return {
        "skill_overlap_ratio": skill_overlap,
        "exp_gap": exp_gap,
        "salary_gap_abs": salary_gap_abs,
        "region_match": region_match,
        "job_popularity_log": job_popularity_log,
        "cand_activity_score": cand_activity_score,
    }


def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-ish batch version used at training time. Delegates to the
    exact same compute_features() per row so there is only one implementation."""
    feats = df.apply(lambda r: compute_features(r), axis=1, result_type="expand")
    return feats[FEATURE_COLUMNS]
