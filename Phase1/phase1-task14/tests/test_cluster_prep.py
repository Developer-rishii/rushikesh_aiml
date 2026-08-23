"""
Tests for Task 14. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_cluster_prep.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from configs.loader import load_config
from src.data.dataset import load_unsupervised_features
from src.clustering.feature_selection import select_features, drop_correlated_features
from src.clustering.prepare import scale_features, apply_pca
from src.clustering.select_k import evaluate_k_range

_CACHE = {}


def _shared_prep():
    if "X_reduced" not in _CACHE:
        cfg = load_config()
        X, y = load_unsupervised_features(cfg)
        selection = select_features(X, cfg)
        X_scaled, scaler = scale_features(selection["selected_features"], cfg)
        pca_result = apply_pca(X_scaled, list(selection["selected_features"].columns), cfg)
        _CACHE.update(cfg=cfg, X=X, y=y, selection=selection, X_scaled=X_scaled,
                       X_reduced=pca_result["X_reduced"], pca_result=pca_result)
    return _CACHE


def test_live_end_to_end_run():
    from src.run_cluster_prep import main
    result = main()
    cfg = load_config()
    assert result["chosen_k"] in cfg.k_range
    assert Path(str(cfg.prepared_data_dir / "clustering_ready_data.csv")).exists()
    total_original = result["n_features_selected"] + result["n_features_dropped_correlated"] + result["n_features_dropped_low_variance"]
    print(f"PASS: live end-to-end run — chosen_k={result['chosen_k']}, "
          f"silhouette={result['chosen_k_silhouette']}, "
          f"{result['n_features_selected']}/{total_original} features kept")


def test_pitfall_features_are_scaled_before_clustering():
    """Pitfall: Clustering on unscaled features."""
    cache = _shared_prep()
    X_scaled = cache["X_scaled"]
    means = np.abs(X_scaled.mean(axis=0))
    stds = X_scaled.std(axis=0)
    assert (means < 1e-6).all(), "scaled features do not have ~zero mean — scaling did not actually run"
    assert np.allclose(stds, 1.0, atol=1e-6), "scaled features do not have unit variance — scaling did not actually run"
    print(f"PASS: all {X_scaled.shape[1]} features confirmed mean~0, std~1 after scaling — "
          f"structurally proven, not just claimed to be scaled")


def test_pitfall_k_is_evidence_based_not_arbitrary():
    """Pitfall: Arbitrary k."""
    cache = _shared_prep()
    cfg = cache["cfg"]
    k_results = evaluate_k_range(cache["X_reduced"], cfg)
    assert len(k_results["per_k_results"]) == len(cfg.k_range), "not every candidate k in the range was evaluated"
    silhouettes = [r["silhouette"] for r in k_results["per_k_results"]]
    assert len(set(silhouettes)) > 1, "all k values produced identical silhouette — evidence isn't discriminating"
    print(f"PASS: k chosen from {len(k_results['per_k_results'])} evaluated candidates via measured silhouette "
          f"scores {silhouettes}, not picked arbitrarily")


def test_pitfall_noisy_dimensions_actually_reduced():
    """Pitfall: Too many noisy dimensions."""
    cache = _shared_prep()
    X, selection, pca_result = cache["X"], cache["selection"], cache["pca_result"]
    assert selection["n_selected"] < X.shape[1], (
        "feature selection kept every original column — redundancy reduction did not actually fire"
    )
    if pca_result["applied"]:
        assert pca_result["n_components"] < selection["n_selected"], (
            "PCA did not actually reduce dimensionality below the selected feature count"
        )
    print(f"PASS: dimensionality actually reduced end-to-end: {X.shape[1]} original -> "
          f"{selection['n_selected']} selected -> {pca_result['n_components']} PCA components")


def test_correlation_dropping_actually_removes_near_duplicates():
    cache = _shared_prep()
    cfg = cache["cfg"]
    X = cache["X"]
    kept, dropped = drop_correlated_features(X, cfg.correlation_threshold)
    assert len(dropped) > 0, "correlation-based dropping found nothing to remove on a dataset with known redundant pairs"
    print(f"PASS: correlation-based redundancy check actually removed {len(dropped)} near-duplicate feature(s)")


def test_edge_case_k_exceeds_sample_size_raises():
    cfg = load_config()
    cfg.k_range = [10000]
    cache = _shared_prep()
    try:
        evaluate_k_range(cache["X_reduced"], cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: requesting k >= n_samples raises clearly instead of crashing inside KMeans")


def test_edge_case_overly_aggressive_selection_raises():
    cfg = load_config()
    cfg.drop_near_zero_variance = True
    cfg.variance_threshold = 1e12  # every feature's variance is "low" -> drops everything
    X, y = load_unsupervised_features(cfg)
    try:
        select_features(X, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: feature-selection thresholds aggressive enough to drop every column raise clearly")


if __name__ == "__main__":
    test_pitfall_features_are_scaled_before_clustering()
    test_pitfall_k_is_evidence_based_not_arbitrary()
    test_pitfall_noisy_dimensions_actually_reduced()
    test_correlation_dropping_actually_removes_near_duplicates()
    test_edge_case_k_exceeds_sample_size_raises()
    test_edge_case_overly_aggressive_selection_raises()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
