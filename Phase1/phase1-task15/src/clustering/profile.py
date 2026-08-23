"""
clustering/profile.py — Steps 3-4: profile each cluster's defining
characteristics on the ORIGINAL, interpretable feature space (never the
PCA components, which have no business meaning), then name clusters in
business terms derived directly from the computed profile — not a
hand-picked label disconnected from the numbers.
"""
import logging
import pandas as pd
import numpy as np

log = logging.getLogger("src.clustering.profile")


def profile_clusters(raw_features: pd.DataFrame, labels: np.ndarray, cfg) -> dict:
    df = raw_features.copy()
    df["_cluster"] = labels
    overall_mean = raw_features.mean()
    overall_std = raw_features.std()

    profiles = {}
    for c in sorted(df["_cluster"].unique()):
        cluster_df = df[df["_cluster"] == c].drop(columns=["_cluster"])
        cluster_mean = cluster_df.mean()
        z_scores = (cluster_mean - overall_mean) / overall_std.replace(0, np.nan)
        top_features = z_scores.abs().sort_values(ascending=False).head(cfg.top_n_defining_features)

        defining = []
        for feat in top_features.index:
            direction = "higher" if z_scores[feat] > 0 else "lower"
            defining.append({
                "feature": feat,
                "cluster_mean": round(float(cluster_mean[feat]), 4),
                "population_mean": round(float(overall_mean[feat]), 4),
                "z_score": round(float(z_scores[feat]), 3),
                "direction": direction,
            })
        profiles[int(c)] = {
            "n_members": int(len(cluster_df)),
            "defining_features": defining,
        }
    log.info("[Step 3] Profiled %s clusters on %s original features each.", len(profiles), raw_features.shape[1])
    return profiles


def name_clusters(profiles: dict) -> dict:
    LABEL_TEMPLATES = {
        "radius": "Tumor Size", "area": "Tumor Size", "perimeter": "Tumor Size",
        "concavity": "Shape Irregularity", "concave points": "Shape Irregularity",
        "compactness": "Shape Irregularity", "smoothness": "Surface Texture",
        "texture": "Surface Texture", "symmetry": "Shape Symmetry",
        "fractal dimension": "Boundary Complexity",
    }
    names = {}
    for cluster_id, profile in profiles.items():
        top = profile["defining_features"][0]
        base_measurement = top["feature"].replace("mean ", "").replace("worst ", "").replace(" error", "")
        theme = next((v for k, v in LABEL_TEMPLATES.items() if k in base_measurement), "Mixed Profile")
        level = "High" if top["direction"] == "higher" else "Low"
        name = f"{level} {theme}"
        names[cluster_id] = {
            "name": name,
            "based_on_feature": top["feature"],
            "z_score": top["z_score"],
        }
    log.info("[Step 4] Named clusters: %s", {k: v["name"] for k, v in names.items()})
    return names
