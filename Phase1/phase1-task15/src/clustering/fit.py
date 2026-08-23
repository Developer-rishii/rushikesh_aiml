"""
clustering/fit.py — Steps 1-2: run K-Means with the locked k, evaluate
quality with silhouette and inertia.
"""
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

log = logging.getLogger("src.clustering.fit")


def run_kmeans(X_reduced: np.ndarray, k: int, seed: int, n_init: int) -> KMeans:
    if k < 2:
        raise ValueError(f"k must be >= 2 for a meaningful clustering, got k={k}.")
    if k >= len(X_reduced):
        raise ValueError(f"k={k} >= n_samples={len(X_reduced)}; cannot cluster.")
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    km.fit(X_reduced)
    log.info("[Step 1] Fit K-Means k=%s (n_init=%s, seed=%s), inertia=%.2f", k, n_init, seed, km.inertia_)
    return km


def evaluate_validity(X_reduced: np.ndarray, labels: np.ndarray) -> dict:
    overall_sil = float(silhouette_score(X_reduced, labels))
    per_sample_sil = silhouette_samples(X_reduced, labels)
    per_cluster_sil = {
        int(c): round(float(per_sample_sil[labels == c].mean()), 4) for c in np.unique(labels)
    }
    cluster_sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    result = {
        "overall_silhouette": round(overall_sil, 4),
        "per_cluster_silhouette": per_cluster_sil,
        "cluster_sizes": cluster_sizes,
    }
    log.info("[Step 2] Validity: %s", result)
    return result
