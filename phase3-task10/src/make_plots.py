"""
make_plots.py
-------------
Turns the JSON artifacts into two evidence plots:
  - offline_metrics.png : baseline vs treatment on nDCG@10 / MAP / precision@5
  - ab_lift_ci.png       : primary metric lift with its 95% CI, against the
                            pre-registered MDE line, so "practical
                            significance" is visible, not just implied.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_offline_metrics():
    with open("artifacts/offline_eval_report.json") as f:
        r = json.load(f)
    metrics = ["nDCG@10", "MAP", "precision@5"]
    baseline_vals = [r["baseline"][m] for m in metrics]
    treatment_vals = [r["treatment"][m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, baseline_vals, w, label="baseline (production heuristic)")
    ax.bar(x + w / 2, treatment_vals, w, label="treatment (learned ranker)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title("Offline eval on held-out queries (never tuned on)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/plots/offline_metrics.png", dpi=150)
    plt.close(fig)


def plot_ab_lift():
    with open("artifacts/experiment_readout.json") as f:
        r = json.load(f)
    with open("artifacts/pre_registration.json") as f:
        pre_reg = json.load(f)

    lift = r["primary_metric"]["relative_lift_pct"]
    ci = r["primary_metric"]["ci95_absolute_diff"]
    control = r["primary_metric"]["control"]
    ci_pct = [c / control * 100 for c in ci] if control else [0, 0]
    mde = pre_reg["minimum_detectable_effect_relative"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([0], [lift], yerr=[[lift - ci_pct[0]], [ci_pct[1] - lift]],
                fmt="o", capsize=8, markersize=10, color="tab:blue", label="observed relative lift")
    ax.axhline(0, color="gray", linewidth=1)
    ax.axhline(mde, color="tab:red", linestyle="--", label=f"pre-registered MDE ({mde:.1f}%)")
    ax.set_xticks([])
    ax.set_ylabel("application_rate relative lift (%)")
    ax.set_title("A/B primary metric: lift with 95% CI vs pre-registered MDE")
    ax.legend()
    fig.tight_layout()
    fig.savefig("artifacts/plots/ab_lift_ci.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_offline_metrics()
    plot_ab_lift()
    print("wrote artifacts/plots/offline_metrics.png and artifacts/plots/ab_lift_ci.png")
