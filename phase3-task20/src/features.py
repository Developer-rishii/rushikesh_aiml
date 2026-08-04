"""
features.py — SINGLE SOURCE OF TRUTH for feature computation.

Guide's core concept: "Train/serve skew ... The single biggest silent killer."
Both train_ranker.py and the serving path (explain.py, latency_bench.py)
import build_features() from here. There is exactly one implementation.
No feature is ever recomputed a second way anywhere else in this repo.
Protected attributes (protected_group) are intentionally excluded from
the feature set -- they are used only in fairness_audit.py, never as
model input.
"""
import pandas as pd


def _overlap(cand_skills, req_skills) -> float:
    # Defensive: malformed/missing skill strings (NaN, None, non-str) are
    # treated as "no skills" rather than crashing the pipeline. Found by
    # the Stage E chaos test (failure_test.py) -- do not remove.
    c = set(str(cand_skills).split(",")) if isinstance(cand_skills, str) else set()
    r = set(str(req_skills).split(",")) if isinstance(req_skills, str) else set()
    return len(c & r) / max(1, len(r))


FEATURE_COLUMNS = ["skill_overlap", "experience_years", "n_candidate_skills", "n_required_skills"]


def build_features(impressions: pd.DataFrame, candidates: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    df = impressions.merge(candidates[["candidate_id", "skills"]], on="candidate_id", how="left")
    df = df.merge(jobs[["job_id", "required_skills"]], on="job_id", how="left")
    df["skill_overlap"] = df.apply(lambda r: _overlap(r["skills"], r["required_skills"]), axis=1)
    df["n_candidate_skills"] = df["skills"].apply(lambda s: len(s.split(",")) if isinstance(s, str) else 0)
    df["n_required_skills"] = df["required_skills"].apply(lambda s: len(s.split(",")) if isinstance(s, str) else 0)
    return df
