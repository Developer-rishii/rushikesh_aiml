"""
clustering/plots.py — a cluster profile bar chart (z-scores of defining
features per cluster) so the profile is visually inspectable, not just
a JSON table.
"""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("src.clustering.plots")


def plot_cluster_profiles(profiles: dict, names: dict, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = len(profiles)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 4), squeeze=False)
    for i, (cluster_id, profile) in enumerate(profiles.items()):
        ax = axes[0][i]
        feats = [d["feature"] for d in profile["defining_features"]]
        zs = [d["z_score"] for d in profile["defining_features"]]
        colors = ["crimson" if z > 0 else "steelblue" for z in zs]
        ax.barh(feats, zs, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Cluster {cluster_id}: {names[cluster_id]['name']} (n={profile['n_members']})")
        ax.set_xlabel("z-score vs population mean")

    fig.tight_layout()
    path = out_dir / "cluster_profiles.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cluster profile plot -> %s", path)
    return str(path)
