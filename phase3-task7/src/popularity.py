import numpy as np

def top_popular(job_df, k=20, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    pool = job_df[~job_df["job_id"].isin(exclude_ids)]
    return pool.sort_values("popularity", ascending=False).head(k)["job_id"].tolist()
