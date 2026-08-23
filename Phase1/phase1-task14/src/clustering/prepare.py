"""
clustering/prepare.py — Steps 2-3: scale so no feature dominates by
unit, then reduce dimensionality if the feature count risks the curse
of dimensionality for Euclidean distance.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

log = logging.getLogger("src.clustering.prepare")


def scale_features(X: pd.DataFrame, cfg) -> tuple:
    if cfg.scaling_method != "standard":
        raise ValueError(f"Unknown scaling method '{cfg.scaling_method}'. Only 'standard' is implemented.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info("[Step 2] Scaled %s features with StandardScaler (mean=0, std=1 each).", X.shape[1])
    return X_scaled, scaler


def apply_pca(X_scaled: np.ndarray, feature_names: list, cfg) -> dict:
    if not cfg.pca_apply:
        return {"applied": False, "X_reduced": X_scaled, "pca": None,
                "n_components": X_scaled.shape[1], "variance_retained": 1.0}

    n_components_cap = min(cfg.pca_max_components, X_scaled.shape[1], X_scaled.shape[0])
    pca_full = PCA(n_components=n_components_cap)
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, cfg.pca_variance_to_retain) + 1)
    n_components = min(n_components, n_components_cap)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    variance_retained = float(np.sum(pca.explained_variance_ratio_))

    log.info("[Step 3] PCA: %s original scaled features -> %s components "
              "(retains %.1f%% variance, target was %.0f%%)",
              X_scaled.shape[1], n_components, variance_retained * 100, cfg.pca_variance_to_retain * 100)
    return {"applied": True, "X_reduced": X_reduced, "pca": pca,
            "n_components": n_components, "variance_retained": round(variance_retained, 4),
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_]}
