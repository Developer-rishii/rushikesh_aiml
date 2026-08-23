"""
clustering/shape_check.py — guards against "Forcing K-Means on
non-spherical data." K-Means assumes round, similar-size clusters
(the guide's own "Limitations" concept) — this checks both assumptions
against what was actually fit, rather than assuming they hold.
"""
import logging
import numpy as np

log = logging.getLogger("src.clustering.shape_check")


def check_kmeans_assumptions(X_reduced: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray) -> dict:
    sizes = np.array([np.sum(labels == c) for c in np.unique(labels)])
    size_ratio = float(sizes.max() / sizes.min())

    spreads = []
    for c in np.unique(labels):
        members = X_reduced[labels == c]
        centroid = cluster_centers[c]
        spread = float(np.mean(np.linalg.norm(members - centroid, axis=1)))
        spreads.append(spread)
    spread_ratio = float(max(spreads) / min(spreads)) if min(spreads) > 0 else float("inf")

    verdict = {
        "cluster_size_ratio": round(size_ratio, 2),
        "cluster_spread_ratio": round(spread_ratio, 2),
        "size_imbalance_flag": size_ratio > 3.0,
        "spread_imbalance_flag": spread_ratio > 2.0,
        "kmeans_assumptions_reasonable": bool(size_ratio <= 3.0 and spread_ratio <= 2.0),
    }
    log.info("K-Means shape-assumption check: %s", verdict)
    return verdict
