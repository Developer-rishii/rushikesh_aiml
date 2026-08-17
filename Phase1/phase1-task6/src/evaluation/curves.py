"""
evaluation/curves.py — Step 4: ROC/PR curve data + plots, showing
performance across ALL thresholds (not just one), before a single
threshold is picked.
"""
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

log = logging.getLogger("src.evaluation.curves")


def compute_curves(y_true, y_proba) -> dict:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist(),
                "auc": round(float(roc_auc_score(y_true, y_proba)), 4)},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(),
               "thresholds": pr_thresholds.tolist(),
               "auc": round(float(average_precision_score(y_true, y_proba)), 4)},
    }


def plot_curves(curves: dict, chosen_threshold: float, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # ROC
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curves["roc"]["fpr"], curves["roc"]["tpr"], label=f"ROC (AUC={curves['roc']['auc']})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (validation)")
    ax.legend()
    roc_path = out_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["roc"] = str(roc_path)

    # PR
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curves["pr"]["recall"], curves["pr"]["precision"], label=f"PR (AUC={curves['pr']['auc']})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (validation)")
    ax.legend()
    pr_path = out_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths["pr"] = str(pr_path)

    log.info("Saved curve plots -> %s, %s", roc_path, pr_path)
    return paths
