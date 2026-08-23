"""
Tests for Task 15. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_kmeans_segments.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from configs.loader import load_config
from src.clustering.fit import run_kmeans
from src.clustering.profile import profile_clusters, name_clusters
from src.clustering.stability import check_stability

_CACHE = {}


def _shared_fit():
    if "labels" not in _CACHE:
        cfg = load_config()
        X_reduced = pd.read_csv(cfg.prepared_data_path).values
        raw_df = pd.read_csv(cfg.raw_data_path)
        raw_features = raw_df[cfg.selected_features]
        km = run_kmeans(X_reduced, cfg.k, cfg.seed, cfg.kmeans_n_init)
        _CACHE.update(cfg=cfg, X_reduced=X_reduced, raw_features=raw_features, km=km, labels=km.labels_)
    return _CACHE["cfg"], _CACHE["X_reduced"], _CACHE["raw_features"], _CACHE["km"], _CACHE["labels"]


def test_live_end_to_end_run():
    from src.run_kmeans_segments import main
    result = main()
    assert len(result["cluster_names"]) == result["k"]
    assert all("recommended_action" in v for v in result["recommendations"].values())
    print(f"PASS: live end-to-end run — k={result['k']}, "
          f"names={[v['name'] for v in result['cluster_names'].values()]}, "
          f"stable={result['stability']['stable']}")


def test_pitfall_clusters_are_interpreted_not_bare_labels():
    """Pitfall: Clusters with no interpretation."""
    cfg, X_reduced, raw_features, km, labels = _shared_fit()
    profiles = profile_clusters(raw_features, labels, cfg)
    names = name_clusters(profiles)
    for cluster_id, profile in profiles.items():
        assert len(profile["defining_features"]) == cfg.top_n_defining_features
        assert names[cluster_id]["name"] != ""
        assert names[cluster_id]["based_on_feature"] == profile["defining_features"][0]["feature"], (
            "cluster name is not actually derived from the computed profile — disconnected label"
        )
    print(f"PASS: every cluster has a computed profile AND a name traceably derived from that "
          f"profile's top feature, not a hand-picked disconnected label")


def test_pitfall_stability_actually_measured():
    """Pitfall: Unstable clusters across runs."""
    cfg, X_reduced, raw_features, km, labels = _shared_fit()
    stability = check_stability(X_reduced, cfg.k, labels, cfg)
    assert len(stability["per_seed_ari"]) == len(cfg.stability_seeds)
    assert all(0 <= r["ari_vs_primary_run"] <= 1.01 for r in stability["per_seed_ari"])
    print(f"PASS: stability actually measured across {len(cfg.stability_seeds)} independent seeds "
          f"(min_ari={stability['min_ari']}), not assumed stable")


def test_pitfall_shape_assumptions_actually_checked():
    """Pitfall: Forcing K-Means on non-spherical data."""
    from src.clustering.shape_check import check_kmeans_assumptions
    cfg, X_reduced, raw_features, km, labels = _shared_fit()
    verdict = check_kmeans_assumptions(X_reduced, labels, km.cluster_centers_)
    assert "cluster_size_ratio" in verdict and "cluster_spread_ratio" in verdict
    assert isinstance(verdict["kmeans_assumptions_reasonable"], bool)
    print(f"PASS: K-Means's round/similar-size assumptions actually checked "
          f"(size_ratio={verdict['cluster_size_ratio']}, spread_ratio={verdict['cluster_spread_ratio']}), "
          f"not assumed to hold")


def test_edge_case_k_too_small_raises():
    cfg, X_reduced, raw_features, km, labels = _shared_fit()
    try:
        run_kmeans(X_reduced, 1, cfg.seed, cfg.kmeans_n_init)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: k=1 (not a meaningful clustering) raises clearly")


def test_edge_case_missing_locked_params_raises():
    from configs.loader import Config
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    raw["data"]["locked_params_path"] = "data/does_not_exist.json"
    try:
        Config(raw, cfg_path)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("PASS: missing Task 14 hand-off (locked_clustering_params.json) raises clearly")


if __name__ == "__main__":
    test_pitfall_clusters_are_interpreted_not_bare_labels()
    test_pitfall_stability_actually_measured()
    test_pitfall_shape_assumptions_actually_checked()
    test_edge_case_k_too_small_raises()
    test_edge_case_missing_locked_params_raises()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
