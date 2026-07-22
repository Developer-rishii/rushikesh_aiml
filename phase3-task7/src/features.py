import numpy as np

def skill_overlap_vector(user_skills, job_df):
    """Jaccard overlap between a user's skills and every job's skills.
    Deterministic, side-effect-free — safe to call identically in training and serving."""
    u = set(user_skills)
    def overlap(job_skills):
        j = set(job_skills)
        union = u | j
        return len(u & j) / len(union) if union else 0.0
    return job_df["skills"].apply(overlap).to_numpy()

def popularity_vector(job_df):
    return job_df["popularity"].to_numpy()

def build_feature_matrix(user_skills, job_df):
    """Returns (n_jobs, 2) matrix: [skill_overlap, popularity]. Extend here, not per-caller,
    so training and serving can never diverge."""
    overlap = skill_overlap_vector(user_skills, job_df)
    pop = popularity_vector(job_df)
    return np.stack([overlap, pop], axis=1)