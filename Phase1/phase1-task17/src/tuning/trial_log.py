"""
tuning/trial_log.py — Step 6: log all trials for reproducibility. Every
trial (completed or pruned) is written to a CSV — the direct guard
against "unreproducible, unlogged trials." Also plots the optimization
history so compute-savings from pruning are visually inspectable.
"""
import logging
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("src.tuning.trial_log")


def save_trial_log(trial_log: list, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in trial_log:
        row = {"trial_number": t["trial_number"], "status": t["status"],
               "folds_completed": t["folds_completed"]}
        row.update(t["params"])
        if t["status"] == "COMPLETE":
            row["score"] = t["mean_cv_score"]
        else:
            row["score"] = t["partial_mean_score"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("trial_number")
    path = out_dir / "all_trials_log.csv"
    df.to_csv(path, index=False)
    log.info("[Step 6] Logged %s trials -> %s", len(df), path)
    return str(path)


def plot_optimization_history(trial_log: list, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete = [t for t in trial_log if t["status"] == "COMPLETE"]
    pruned = [t for t in trial_log if t["status"] == "PRUNED"]

    fig, ax = plt.subplots(figsize=(8, 5))
    if complete:
        ax.scatter([t["trial_number"] for t in complete], [t["mean_cv_score"] for t in complete],
                    color="steelblue", label=f"completed ({len(complete)})", zorder=3)
        running_best = []
        best_so_far = -1
        for t in sorted(complete, key=lambda x: x["trial_number"]):
            best_so_far = max(best_so_far, t["mean_cv_score"])
            running_best.append((t["trial_number"], best_so_far))
        ax.plot([p[0] for p in running_best], [p[1] for p in running_best],
                 color="darkgreen", linestyle="--", label="running best")
    if pruned:
        ax.scatter([t["trial_number"] for t in pruned], [t["partial_mean_score"] for t in pruned],
                    color="crimson", marker="x", label=f"pruned early ({len(pruned)})", zorder=3)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV score (PR-AUC)")
    ax.set_title("Bayesian search: optimization history (pruned trials marked)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "optimization_history.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved optimization history plot -> %s", path)
    return str(path)
