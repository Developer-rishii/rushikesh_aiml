from src.features import build_feature_matrix, skill_overlap_vector, popularity_vector
import numpy as np

def check_skew(user_skills_list, job_df, tolerance=1e-9):
    """
    Validates train/serve feature consistency across a sample of user skill profiles.
    Ensures zero feature divergence (max_diff < tolerance) between offline pipeline and serving layer.
    """
    max_diffs = []
    if isinstance(user_skills_list[0], str):
        user_skills_list = [user_skills_list]

    for skills in user_skills_list:
        train_feats = build_feature_matrix(skills, job_df)
        # simulate serving feature computation
        overlap_serve = skill_overlap_vector(skills, job_df)
        pop_serve = popularity_vector(job_df)
        serve_feats = np.stack([overlap_serve, pop_serve], axis=1)

        diff = np.abs(train_feats - serve_feats).max()
        max_diffs.append(diff)

    overall_max_diff = float(np.max(max_diffs))
    assert overall_max_diff < tolerance, f"Train/serve skew detected: {overall_max_diff}"

    return {
        "max_diff": overall_max_diff,
        "status": "PASSED_ZERO_SKEW",
        "tolerance": tolerance,
        "samples_checked": len(user_skills_list)
    }