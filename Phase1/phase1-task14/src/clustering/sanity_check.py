"""
clustering/sanity_check.py — Step 5: sanity-check that distances are
meaningful. Two checks:
  1. Silhouette at the chosen k must clear a minimum bar — a silhouette
     near 0 means clusters are not actually separated, i.e. distances
     aren't finding real structure (curse-of-dimensionality symptom).
  2. Distance concentration: ratio of max to min pairwise distance in a
     sample — as dimensionality rises this ratio collapses toward 1,
     meaning "nearest" and "farthest" points become indistinguishable
     (the textbook curse-of-dimensionality failure mode this task's
     concepts section names directly).
"""
import logging
import numpy as np
from scipy.spatial.distance import pdist

log = logging.getLogger("src.clustering.sanity_check")


def check_distance_meaningfulness(X_reduced: np.ndarray, chosen_k: int, chosen_silhouette: float, cfg) -> dict:
    if len(X_reduced) <= 300:
        sample = X_reduced
    else:
        idx = np.random.default_rng(cfg.seed).choice(len(X_reduced), 300, replace=False)
        sample = X_reduced[idx]
    distances = pdist(sample, metric="euclidean")
    ratio = float(distances.max() / distances.min()) if distances.min() > 0 else float("inf")

    silhouette_ok = chosen_silhouette >= cfg.min_acceptable_silhouette
    distance_concentration_flag = ratio < 2.0

    verdict = {
        "chosen_k": chosen_k,
        "chosen_silhouette": chosen_silhouette,
        "min_acceptable_silhouette": cfg.min_acceptable_silhouette,
        "silhouette_check_passed": silhouette_ok,
        "max_min_distance_ratio": round(ratio, 2),
        "distance_concentration_flagged": distance_concentration_flag,
        "distances_meaningful": bool(silhouette_ok and not distance_concentration_flag),
    }
    log.info("[Step 5] Distance sanity check: %s", verdict)
    return verdict
