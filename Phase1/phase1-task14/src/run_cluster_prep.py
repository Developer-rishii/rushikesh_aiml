"""
run_cluster_prep.py — Task 14's full flow, in the study guide's exact
step order:
  1. Select the features that should define the segments.
  2. Scale them so none dominates by unit.
  3. Reduce dimensionality if needed (PCA).
  4. Use elbow/silhouette to choose a candidate k.
  5. Sanity-check that distances are meaningful.
  6. Lock the prepared dataset and chosen parameters.

Run: python -m src.run_cluster_prep
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib

from configs.loader import load_config
from src.data.dataset import load_unsupervised_features
from src.clustering.feature_selection import select_features
from src.clustering.prepare import scale_features, apply_pca
from src.clustering.select_k import evaluate_k_range, plot_elbow_and_silhouette
from src.clustering.sanity_check import check_distance_meaningfulness

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_cluster_prep")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    try:
        X, y_reference = load_unsupervised_features(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    try:
        selection = select_features(X, cfg)
    except ValueError as e:
        log.error("Feature selection failed: %s", e)
        sys.exit(1)
    X_selected = selection["selected_features"]
    log.info("[Step 1] Selected %s/%s features: %s", selection["n_selected"], selection["n_original"],
              list(X_selected.columns))

    try:
        X_scaled, scaler = scale_features(X_selected, cfg)
    except ValueError as e:
        log.error("Scaling failed: %s", e)
        sys.exit(1)

    pca_result = apply_pca(X_scaled, list(X_selected.columns), cfg)
    X_reduced = pca_result["X_reduced"]

    try:
        k_results = evaluate_k_range(X_reduced, cfg)
    except ValueError as e:
        log.error("k-selection failed: %s", e)
        sys.exit(1)
    plot_path = plot_elbow_and_silhouette(k_results, cfg.figure_dir)
    chosen_k = k_results["best_k_by_silhouette"]

    sanity = check_distance_meaningfulness(X_reduced, chosen_k, k_results["best_silhouette"], cfg)
    if not sanity["distances_meaningful"]:
        log.warning("Distance sanity check FAILED — chosen k/features may not produce meaningful clusters. "
                     "Proceeding to lock anyway, but this is flagged prominently in the report.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    km_final = KMeans(n_clusters=chosen_k, n_init=cfg.k_n_init, random_state=cfg.seed)
    cluster_labels = km_final.fit_predict(X_reduced)
    ari_vs_diagnosis = round(float(adjusted_rand_score(y_reference, cluster_labels)), 4)
    log.info("[Step 5b, informational] Adjusted Rand Index vs the (unused-in-prep) diagnosis label: %s "
              "— not used to choose features or k, reported only as an external sanity reference.",
              ari_vs_diagnosis)

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.prepared_data_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = pd.DataFrame(X_reduced, columns=[f"pc_{i+1}" for i in range(X_reduced.shape[1])])
    prepared_df.to_csv(cfg.prepared_data_dir / "clustering_ready_data.csv", index=False)
    joblib.dump(scaler, cfg.prepared_data_dir / "scaler.joblib")
    if pca_result["pca"] is not None:
        joblib.dump(pca_result["pca"], cfg.prepared_data_dir / "pca.joblib")

    locked_params = {
        "seed": cfg.seed,
        "selected_features": list(X_selected.columns),
        "n_features_selected": selection["n_selected"],
        "n_features_dropped_low_variance": len(selection["low_variance_dropped"]),
        "n_features_dropped_correlated": len(selection["correlation_dropped"]),
        "scaling_method": cfg.scaling_method,
        "pca_applied": pca_result["applied"],
        "pca_n_components": pca_result["n_components"],
        "pca_variance_retained": pca_result["variance_retained"],
        "chosen_k": chosen_k,
        "chosen_k_silhouette": k_results["best_silhouette"],
        "k_selection_evidence": k_results["per_k_results"],
        "distance_sanity_check": sanity,
        "external_reference_ari_vs_diagnosis": ari_vs_diagnosis,
    }
    (cfg.prepared_data_dir / "locked_clustering_params.json").write_text(json.dumps(locked_params, indent=2))

    result = {
        **locked_params,
        "n_rows": X.shape[0],
        "feature_selection_detail": {
            "low_variance_dropped": selection["low_variance_dropped"],
            "correlation_dropped": selection["correlation_dropped"],
        },
        "elbow_silhouette_plot": plot_path,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "cluster_prep_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_cluster_prep.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("[Step 6] Locked prepared dataset + params -> %s", cfg.prepared_data_dir)
    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], cfg.report_dir / "cluster_prep_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
