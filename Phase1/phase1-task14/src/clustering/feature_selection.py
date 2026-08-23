"""
clustering/feature_selection.py — Step 1: select the features that
should define the segments, via redundancy reduction (not target-based
importance — clustering is unsupervised, using the label here would be
a supervised shortcut). Two passes:
  1. Drop near-zero-variance columns (carry ~no distinguishing signal).
  2. Drop features highly correlated (>threshold) with an already-kept
     feature — this directly targets "too many noisy/redundant
     dimensions" and the specific WDBC structure where mean/worst pairs
     for the same measurement (e.g. mean radius vs worst radius) are
     near-duplicates that would otherwise let one physical property
     dominate distance 2-3x over.
"""
import logging
import pandas as pd

log = logging.getLogger("src.clustering.feature_selection")


def drop_low_variance(X: pd.DataFrame, threshold: float) -> tuple:
    """
    Uses coefficient of variation (std / |mean|), NOT raw variance,
    because WDBC's raw features span wildly different absolute scales
    (e.g. 'mean area' ~ hundreds to thousands vs 'mean smoothness' ~
    0.05-0.16). An absolute-variance cutoff would flag small-scale-but-
    genuinely-informative features (smoothness, symmetry, fractal
    dimension) as "near-constant" purely because of their unit, not
    because they lack signal -- exactly the kind of scale-driven
    distortion this task's Step 2 exists to prevent, so it must not
    leak into Step 1's selection logic either.
    """
    means = X.mean().abs()
    stds = X.std()
    coeff_variation = stds / (means + 1e-9)
    low_var_cols = coeff_variation[coeff_variation < threshold].index.tolist()
    kept = X.drop(columns=low_var_cols)
    if low_var_cols:
        log.info("[Step 1a] Dropped %s near-zero-(relative)-variance feature(s) "
                  "(coefficient of variation < %s): %s", len(low_var_cols), threshold, low_var_cols)
    return kept, low_var_cols


def drop_correlated_features(X: pd.DataFrame, threshold: float) -> tuple:
    """Greedy: walk features in a fixed order, drop any feature whose
    absolute correlation with an ALREADY-KEPT feature exceeds threshold."""
    corr = X.corr().abs()
    kept_cols = []
    dropped = {}
    for col in X.columns:
        is_redundant = False
        for kept_col in kept_cols:
            if corr.loc[col, kept_col] > threshold:
                dropped[col] = {"redundant_with": kept_col, "correlation": round(float(corr.loc[col, kept_col]), 4)}
                is_redundant = True
                break
        if not is_redundant:
            kept_cols.append(col)
    log.info("[Step 1b] Dropped %s correlated feature(s) (threshold=%s): %s",
              len(dropped), threshold, list(dropped.keys()))
    return X[kept_cols], dropped


def select_features(X: pd.DataFrame, cfg) -> dict:
    working = X.copy()
    low_var_dropped = []
    if cfg.drop_near_zero_variance:
        working, low_var_dropped = drop_low_variance(working, cfg.variance_threshold)

    working, corr_dropped = drop_correlated_features(working, cfg.correlation_threshold)

    if working.shape[1] == 0:
        raise ValueError("Feature selection dropped every column — thresholds are too aggressive.")

    return {
        "selected_features": working,
        "low_variance_dropped": low_var_dropped,
        "correlation_dropped": corr_dropped,
        "n_original": X.shape[1],
        "n_selected": working.shape[1],
    }
