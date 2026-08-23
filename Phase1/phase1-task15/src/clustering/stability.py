"""
clustering/stability.py — Step 5: check stability across seeds. Refits
K-Means with several different random seeds and measures agreement
(Adjusted Rand Index) against the primary run's labels — the direct,
measured guard against "unstable clusters across runs."
"""
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

log = logging.getLogger("src.clustering.stability")


def check_stability(X_reduced: np.ndarray, k: int, primary_labels: np.ndarray, cfg) -> dict:
    aris = []
    for seed in cfg.stability_seeds:
        km = KMeans(n_clusters=k, n_init=cfg.kmeans_n_init, random_state=seed)
        labels = km.fit_predict(X_reduced)
        ari = adjusted_rand_score(primary_labels, labels)
        aris.append({"seed": seed, "ari_vs_primary_run": round(float(ari), 4)})

    aris_only = [r["ari_vs_primary_run"] for r in aris]
    mean_ari = float(np.mean(aris_only))
    min_ari = float(np.min(aris_only))
    stable = min_ari >= cfg.min_acceptable_ari

    result = {
        "seeds_checked": cfg.stability_seeds,
        "per_seed_ari": aris,
        "mean_ari": round(mean_ari, 4),
        "min_ari": round(min_ari, 4),
        "min_acceptable_ari": cfg.min_acceptable_ari,
        "stable": bool(stable),
    }
    log.info("[Step 5] Stability across %s seeds: mean_ari=%.4f min_ari=%.4f -> %s",
              len(cfg.stability_seeds), mean_ari, min_ari, "STABLE" if stable else "UNSTABLE")
    return result
