"""
Train/serve skew is called out in the study guide as "the single biggest silent
killer". The fix: ONE function computes candidate-job pair features, imported
by both train.py (offline) and serve.py (online). Never duplicate this logic.
"""
import numpy as np
import pandas as pd

SKILLS = [f"skill_{i}" for i in range(25)]
LEVELS = ["intern", "junior", "mid", "senior", "lead"]


def pair_features(cand_row: pd.Series, job_row: pd.Series) -> dict:
    c_vec = cand_row[SKILLS].values.astype(float)
    j_vec = job_row[SKILLS].values.astype(float)
    skill_overlap = float(np.dot(c_vec, j_vec))
    skill_overlap_norm = skill_overlap / (np.sqrt(j_vec.sum()) + 1e-6)
    same_city = float(cand_row["city"] == job_row["city"] or job_row["city"] == "Remote")
    level_dist = abs(LEVELS.index(cand_row["level"]) - LEVELS.index(job_row["level"]))
    years_exp = float(cand_row["years_exp"])
    popularity_prior = float(job_row["popularity_prior"])
    company_pref_weight = float(job_row["company_pref_weight"])
    return {
        "skill_overlap": skill_overlap,
        "skill_overlap_norm": skill_overlap_norm,
        "same_city": same_city,
        "level_dist": level_dist,
        "years_exp": years_exp,
        "popularity_prior": popularity_prior,
        "company_pref_weight": company_pref_weight,
    }


FEATURE_COLUMNS = [
    "skill_overlap", "skill_overlap_norm", "same_city",
    "level_dist", "years_exp", "popularity_prior", "company_pref_weight",
]


def build_feature_frame(pairs_df, candidates_df, jobs_df):
    """pairs_df needs columns candidate_id, job_id. Returns feature matrix aligned to pairs_df rows."""
    cand_idx = candidates_df.set_index("candidate_id")
    job_idx = jobs_df.set_index("job_id")
    feats = []
    for _, r in pairs_df.iterrows():
        c = cand_idx.loc[r["candidate_id"]]
        j = job_idx.loc[r["job_id"]]
        feats.append(pair_features(c, j))
    return pd.DataFrame(feats, columns=FEATURE_COLUMNS)
