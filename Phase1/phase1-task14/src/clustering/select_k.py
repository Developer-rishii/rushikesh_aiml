"""
clustering/select_k.py — Step 4: use elbow (inertia) + silhouette to
choose a candidate k. Both computed and reported together — silhouette
is the primary justification (elbow is famously ambiguous to read
mechanically), but both are shown so the choice is a documented,
evidence-based pick, not "whatever number looked convenient."
"""
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

log = logging.getLogger("src.clustering.select_k")


def evaluate_k_range(X_reduced: np.ndarray, cfg) -> dict:
    results = []
    for k in cfg.k_range:
        if k >= X_reduced.shape[0]:
            raise ValueError(f"k={k} is >= n_samples={X_reduced.shape[0]}; cannot cluster.")
        km = KMeans(n_clusters=k, n_init=cfg.k_n_init, random_state=cfg.seed)
        labels = km.fit_predict(X_reduced)
        sil = silhouette_score(X_reduced, labels)
        results.append({"k": k, "inertia": round(float(km.inertia_), 2), "silhouette": round(float(sil), 4)})
        log.info("[Step 4] k=%s: inertia=%.2f, silhouette=%.4f", k, km.inertia_, sil)

    best = max(results, key=lambda r: r["silhouette"])
    return {"per_k_results": results, "best_k_by_silhouette": best["k"], "best_silhouette": best["silhouette"]}


def plot_elbow_and_silhouette(k_results: dict, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = k_results["per_k_results"]
    ks = [r["k"] for r in rows]
    inertias = [r["inertia"] for r in rows]
    sils = [r["silhouette"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ks, inertias, marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster SS)")
    ax1.set_title("Elbow method")

    ax2.plot(ks, sils, marker="o", color="darkorange")
    ax2.axvline(k_results["best_k_by_silhouette"], linestyle="--", color="gray",
                label=f"chosen k={k_results['best_k_by_silhouette']}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette method")
    ax2.legend()

    fig.tight_layout()
    path = out_dir / "elbow_silhouette.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("[Step 4] Saved elbow+silhouette plot -> %s", path)
    return str(path)
