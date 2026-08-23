"""
run_kmeans_segments.py — Task 15's full flow, in the study guide's exact
step order:
  1. Run K-Means with the chosen k on prepared data.
  2. Evaluate quality with silhouette and inertia.
  3. Profile each cluster's defining characteristics.
  4. Name clusters in business terms.
  5. Check stability across seeds.
  6. Recommend an action per segment.

Run: python -m src.run_kmeans_segments
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
from src.clustering.fit import run_kmeans, evaluate_validity
from src.clustering.profile import profile_clusters, name_clusters
from src.clustering.stability import check_stability
from src.clustering.shape_check import check_kmeans_assumptions
from src.clustering.recommend import recommend_actions
from src.clustering.plots import plot_cluster_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_kmeans_segments")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s (k=%s locked from Task 14, justified by silhouette=%s)",
              cfg, cfg.k, cfg.k_justification_silhouette)

    try:
        if not cfg.prepared_data_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {cfg.prepared_data_path}")
        X_reduced = pd.read_csv(cfg.prepared_data_path).values
        if not cfg.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found: {cfg.raw_data_path}")
        raw_df = pd.read_csv(cfg.raw_data_path)
        y_reference = raw_df[cfg.target_col]
        raw_features = raw_df[cfg.selected_features] if cfg.selected_features else raw_df.drop(columns=[cfg.target_col])
    except FileNotFoundError as e:
        log.error("Data hand-off from Task 14 failed: %s", e)
        sys.exit(1)

    if len(X_reduced) != len(raw_features):
        log.error("Row count mismatch between prepared data (%s) and raw data (%s) — hand-off inconsistency.",
                   len(X_reduced), len(raw_features))
        sys.exit(1)

    try:
        km = run_kmeans(X_reduced, cfg.k, cfg.seed, cfg.kmeans_n_init)
    except ValueError as e:
        log.error("K-Means fit failed: %s", e)
        sys.exit(1)
    labels = km.labels_

    validity = evaluate_validity(X_reduced, labels)

    shape_check = check_kmeans_assumptions(X_reduced, labels, km.cluster_centers_)
    if not shape_check["kmeans_assumptions_reasonable"]:
        log.warning("K-Means's round/similar-size assumption looks shaky for this data: %s", shape_check)

    profiles = profile_clusters(raw_features, labels, cfg)
    names = name_clusters(profiles)
    plot_path = plot_cluster_profiles(profiles, names, cfg.figure_dir)
    stability = check_stability(X_reduced, cfg.k, labels, cfg)
    recommendations = recommend_actions(profiles, names, labels, y_reference)

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, cfg.artifact_dir / "kmeans_model.joblib")

    labeled_df = raw_features.copy()
    labeled_df["cluster"] = labels
    labeled_df["cluster_name"] = [names[c]["name"] for c in labels]
    labeled_df.to_csv(cfg.artifact_dir / "labeled_records.csv", index=False)

    result = {
        "seed": cfg.seed,
        "k": cfg.k,
        "k_justification_from_task14": {"silhouette": cfg.k_justification_silhouette},
        "n_rows": len(X_reduced),
        "validity": validity,
        "shape_assumption_check": shape_check,
        "cluster_profiles": profiles,
        "cluster_names": names,
        "stability": stability,
        "recommendations": recommendations,
        "cluster_profile_plot": plot_path,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "kmeans_segments_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_kmeans_segments.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], cfg.report_dir / "kmeans_segments_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
