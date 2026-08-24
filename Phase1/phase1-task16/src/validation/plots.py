"""validation/plots.py — box/strip plot of per-fold scores per model."""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("src.validation.plots")


def plot_fold_scores(per_model_results: dict, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(per_model_results.keys())
    data = [per_model_results[n]["fold_scores"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=names, showmeans=True)
    for i, scores in enumerate(data, start=1):
        ax.scatter([i] * len(scores), scores, alpha=0.6, color="darkorange", zorder=3)
    ax.set_ylabel("Fold score")
    ax.set_title("Per-fold CV scores by model (mean + spread, not just best fold)")
    fig.tight_layout()
    path = out_dir / "fold_scores_comparison.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fold-score comparison plot -> %s", path)
    return str(path)
