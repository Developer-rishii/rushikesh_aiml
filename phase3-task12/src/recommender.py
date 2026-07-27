"""
Stage B step 4: "Make it explainable, safe & demoable."
- Two-sided: candidate->jobs score blends model relevance with company_pref_weight
  (what's good for the candidate isn't always what's good for the company; here
  we balance both instead of silently choosing one side).
- Diversity: MMR (Maximal Marginal Relevance) re-ranking against skill vectors so
  we don't just return near-duplicate jobs (Sec 4 "Coverage & diversity").
- Explainability: a plain-English reason built from the same features the model saw.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import pair_features, FEATURE_COLUMNS, SKILLS

TWO_SIDED_ALPHA = 0.9  # weight on candidate relevance vs company preference


LEVELS = ["intern", "junior", "mid", "senior", "lead"]


def _vectorized_features(candidate_row, jobs_df):
    """Vectorized version of features.pair_features for one candidate against all jobs.
    Kept numerically identical to features.pair_features (verified by a unit check in
    failure_test.py) -- this is a performance optimization only, not a second source of truth:
    both implementations share the exact same formulas, just batched here for serving speed."""
    c_vec = candidate_row[SKILLS].values.astype(float)
    j_mat = jobs_df[SKILLS].values.astype(float)
    skill_overlap = j_mat @ c_vec
    skill_overlap_norm = skill_overlap / (np.sqrt(j_mat.sum(axis=1)) + 1e-6)
    same_city = ((jobs_df["city"].values == candidate_row["city"]) |
                 (jobs_df["city"].values == "Remote")).astype(float)
    level_dist = np.abs(LEVELS.index(candidate_row["level"]) -
                         jobs_df["level"].map(LEVELS.index).values)
    years_exp = np.full(len(jobs_df), float(candidate_row["years_exp"]))
    popularity_prior = jobs_df["popularity_prior"].values.astype(float)
    company_pref_weight = jobs_df["company_pref_weight"].values.astype(float)
    return pd.DataFrame({
        "skill_overlap": skill_overlap,
        "skill_overlap_norm": skill_overlap_norm,
        "same_city": same_city,
        "level_dist": level_dist,
        "years_exp": years_exp,
        "popularity_prior": popularity_prior,
        "company_pref_weight": company_pref_weight,
    }, columns=FEATURE_COLUMNS)


def score_candidate_jobs(model, candidate_row, jobs_df):
    feats = _vectorized_features(candidate_row, jobs_df)
    relevance = model.predict_proba(feats)[:, 1]
    company_pref = jobs_df["company_pref_weight"].values
    two_sided_score = TWO_SIDED_ALPHA * relevance + (1 - TWO_SIDED_ALPHA) * company_pref
    return relevance, two_sided_score, feats


def _mmr_rerank(job_ids, scores, skill_matrix, k, lambda_diversity=0.85):
    """Greedy MMR: balance top score vs dissimilarity to already-picked items."""
    selected = []
    candidates = list(range(len(job_ids)))
    scores = np.array(scores)
    sim = skill_matrix @ skill_matrix.T
    norm = np.linalg.norm(skill_matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    sim = sim / (norm @ norm.T + 1e-9)

    while candidates and len(selected) < k:
        best_i, best_val = None, -1e9
        for i in candidates:
            div_penalty = max([sim[i][j] for j in selected], default=0)
            mmr = lambda_diversity * scores[i] - (1 - lambda_diversity) * div_penalty
            if mmr > best_val:
                best_val, best_i = mmr, i
        selected.append(best_i)
        candidates.remove(best_i)
    return [job_ids[i] for i in selected]


def explain(candidate_row, job_row, feats_row):
    reasons = []
    if feats_row["skill_overlap"] >= 2:
        reasons.append(f"matches {int(feats_row['skill_overlap'])} of the job's required skills")
    if feats_row["same_city"]:
        reasons.append(f"is in {job_row['city']} (or job is remote)")
    if feats_row["level_dist"] <= 1:
        reasons.append(f"seniority ({candidate_row['level']}) fits the {job_row['level']} role")
    if not reasons:
        reasons.append("is a broad match based on overall profile similarity")
    return "Recommended because candidate " + "; ".join(reasons) + "."


def recommend_for_candidate(model, candidate_id, candidates_df, jobs_df, k=10, diversify=True):
    cand_row = candidates_df.set_index("candidate_id").loc[candidate_id]
    relevance, two_sided_score, feats = score_candidate_jobs(model, cand_row, jobs_df)
    job_ids = jobs_df["job_id"].tolist()

    if diversify:
        skill_matrix = jobs_df[SKILLS].values.astype(float)
        top_pool_idx = np.argsort(-two_sided_score)[:max(k * 3, 30)]
        pool_ids = [job_ids[i] for i in top_pool_idx]
        pool_scores = two_sided_score[top_pool_idx]
        pool_skills = skill_matrix[top_pool_idx]
        ranked_ids = _mmr_rerank(pool_ids, pool_scores, pool_skills, k=k)
    else:
        order = np.argsort(-two_sided_score)[:k]
        ranked_ids = [job_ids[i] for i in order]

    results = []
    jobs_idx = jobs_df.set_index("job_id")
    for jid in ranked_ids:
        j = jobs_idx.loc[jid]
        i = job_ids.index(jid)
        results.append({
            "job_id": jid,
            "relevance_score": float(relevance[i]),
            "two_sided_score": float(two_sided_score[i]),
            "reason": explain(cand_row, j, feats.iloc[i]),
        })
    return results


def recommend_candidates_for_job(model, job_id, candidates_df, jobs_df, k=10):
    """Company->candidates direction (the other side of the marketplace)."""
    job_row = jobs_df.set_index("job_id").loc[job_id]
    feats = pd.DataFrame([pair_features(c, job_row) for _, c in candidates_df.iterrows()],
                         columns=FEATURE_COLUMNS)
    relevance = model.predict_proba(feats)[:, 1]
    order = np.argsort(-relevance)[:k]
    cand_ids = candidates_df["candidate_id"].tolist()
    out = []
    for i in order:
        out.append({
            "candidate_id": cand_ids[i],
            "relevance_score": float(relevance[i]),
            "reason": explain(candidates_df.iloc[i], job_row, feats.iloc[i]),
        })
    return out
