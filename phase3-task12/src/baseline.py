"""
Stage B step 1: "Write down the baseline you must beat."
Baseline = pure popularity ranking, no personalization. This is the naive
in-production ranker our synthetic logs were partly generated from. Any real
model must beat this on offline metrics, or it isn't earning its complexity.
"""
import pandas as pd


def popularity_baseline_topk(jobs_df: pd.DataFrame, k: int = 10):
    ranked = jobs_df.sort_values("popularity_prior", ascending=False)
    return ranked["job_id"].tolist()[:k]


def recommend_for_all_candidates_baseline(candidates_df, jobs_df, k=10):
    topk = popularity_baseline_topk(jobs_df, k=k)
    return {cid: topk for cid in candidates_df["candidate_id"]}
