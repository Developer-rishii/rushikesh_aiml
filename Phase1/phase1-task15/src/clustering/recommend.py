"""
clustering/recommend.py — Step 6: recommend an action per segment,
derived from the cluster's profile and its (informational-only) overlap
with the actual diagnosis label — never used to form the clusters, only
to make the recommendation concrete and checkable.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("src.clustering.recommend")


def recommend_actions(profiles: dict, names: dict, labels: np.ndarray, y_reference: pd.Series) -> dict:
    recommendations = {}
    for cluster_id, profile in profiles.items():
        mask = labels == cluster_id
        ref = y_reference[mask]
        malignant_rate = float((ref == 0).mean()) if len(ref) else None
        top = profile["defining_features"][0]

        if malignant_rate is not None and malignant_rate > 0.5:
            action = (
                "PRIORITY REVIEW: this segment skews malignant "
                f"({malignant_rate:.0%} of members) and is defined by {top['direction']} "
                f"{top['feature']} — route for expedited pathologist follow-up rather than "
                "standard-timeline review."
            )
        elif malignant_rate is not None:
            action = (
                "STANDARD MONITORING: this segment skews benign "
                f"({1-malignant_rate:.0%} of members) and is defined by {top['direction']} "
                f"{top['feature']} — appropriate for routine follow-up scheduling, "
                "not expedited review."
            )
        else:
            action = "No reference label available — recommend manual clinical review to characterize this segment."

        recommendations[cluster_id] = {
            "cluster_name": names[cluster_id]["name"],
            "n_members": profile["n_members"],
            "malignant_rate_in_segment": round(malignant_rate, 4) if malignant_rate is not None else None,
            "recommended_action": action,
        }
    log.info("[Step 6] Recommendations: %s", {k: v["cluster_name"] for k, v in recommendations.items()})
    return recommendations
