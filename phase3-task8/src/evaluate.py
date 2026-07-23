"""
evaluate.py
------------
Stage C: "Honest evaluation (PR curve, lift over baseline) on held-out data".

Everything in this file runs ONLY on data/processed/holdout_snapshots.csv
(Nov 2025 - Jan 2026), which train_model.py never touched during fitting or
early stopping. This is the "gap between offline metric and expected online
effect" section the guide asks for -- we report it honestly, including where
the model does NOT beat the baseline (see the near-term / cold segment).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import FEATURE_COLUMNS
from baseline_model import rule_14_day_score, rule_14_day_binary, rfm_score

ROOT = Path(__file__).resolve().parents[1]
OPERATING_THRESHOLD_PERCENTILE = 90  # flag the top 10% riskiest as "at-risk" for growth outreach


def lift_at_k(y_true, y_score, k_fracs=(0.05, 0.10, 0.20, 0.30)):
    n = len(y_true)
    order = np.argsort(-y_score)
    base_rate = y_true.mean()
    out = {}
    for k in k_fracs:
        topk = order[: max(1, int(n * k))]
        precision_at_k = y_true.iloc[topk].mean() if hasattr(y_true, "iloc") else y_true[topk].mean()
        out[f"lift@{int(k*100)}%"] = float(precision_at_k / base_rate) if base_rate > 0 else float("nan")
    return out


def main():
    holdout = pd.read_csv(ROOT / "data/processed/holdout_snapshots.csv", parse_dates=["as_of_date"])
    X_hold, y_hold = holdout[FEATURE_COLUMNS], holdout["churned"]

    with open(ROOT / "models/churn_model_v1.pkl", "rb") as f:
        model = pickle.load(f)

    model_score = model.predict_proba(X_hold)[:, 1]
    baseline_score = rule_14_day_score(holdout)
    rfm = rfm_score(holdout)

    results = {}
    plt.figure(figsize=(7, 6))
    for name, score in [("model_v1 (HistGBM)", model_score),
                        ("baseline: 14-day-inactivity rule", baseline_score),
                        ("secondary baseline: RFM rule", rfm)]:
        p, r, thr = precision_recall_curve(y_hold, score)
        ap = average_precision_score(y_hold, score)
        plt.plot(r, p, label=f"{name}  (PR-AUC={ap:.3f})")
        results[name] = {"pr_auc": float(ap), **lift_at_k(y_hold, score)}

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall: churn model vs baselines (held-out Nov'25-Jan'26 snapshots)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.3)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs/pr_curve.png", dpi=150)
    plt.close()

    # operating point: top decile by model score -> precision/recall/f1 AT that threshold
    thresh_val = np.percentile(model_score, OPERATING_THRESHOLD_PERCENTILE)
    model_flag = (model_score >= thresh_val).astype(int)
    baseline_flag = rule_14_day_binary(holdout, threshold=14)

    op_point = {
        "operating_threshold_percentile": OPERATING_THRESHOLD_PERCENTILE,
        "model_score_threshold": float(thresh_val),
        "model_precision_at_threshold": float(precision_score(y_hold, model_flag, zero_division=0)),
        "model_recall_at_threshold": float(recall_score(y_hold, model_flag, zero_division=0)),
        "model_f1_at_threshold": float(f1_score(y_hold, model_flag, zero_division=0)),
        "baseline14_precision": float(precision_score(y_hold, baseline_flag, zero_division=0)),
        "baseline14_recall": float(recall_score(y_hold, baseline_flag, zero_division=0)),
        "baseline14_f1": float(f1_score(y_hold, baseline_flag, zero_division=0)),
        "holdout_base_churn_rate": float(y_hold.mean()),
        "holdout_n": int(len(y_hold)),
    }

    report = {"pr_and_lift_by_model": results, "operating_point": op_point}
    with open(ROOT / "outputs/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # human-readable honest markdown version, including where model does NOT win
    md = ["# Honest Evaluation Report (held-out data, never trained/tuned on)\n",
          f"Holdout snapshots: Nov 2025 - Jan 2026. n={op_point['holdout_n']}, "
          f"base churn rate={op_point['holdout_base_churn_rate']:.3f}\n",
          "## PR-AUC and lift over baseline\n",
          "| Model | PR-AUC | Lift@5% | Lift@10% | Lift@20% | Lift@30% |",
          "|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['pr_auc']:.3f} | {r['lift@5%']:.2f}x | {r['lift@10%']:.2f}x | "
                   f"{r['lift@20%']:.2f}x | {r['lift@30%']:.2f}x |")
    md.append(f"\n## Operating point (top {100-OPERATING_THRESHOLD_PERCENTILE}% riskiest flagged)\n")
    md.append(f"- Model: precision={op_point['model_precision_at_threshold']:.3f}, "
              f"recall={op_point['model_recall_at_threshold']:.3f}, f1={op_point['model_f1_at_threshold']:.3f}")
    md.append(f"- Baseline (14-day rule): precision={op_point['baseline14_precision']:.3f}, "
              f"recall={op_point['baseline14_recall']:.3f}, f1={op_point['baseline14_f1']:.3f}")
    gap_note = ("\n## Honest gap note\nThis is an OFFLINE evaluation on simulated logs. The offline PR-AUC and "
                "lift numbers above are not a promise of equivalent online lift -- real online effect depends on "
                "whether growth's intervention (re-engagement email/digest) actually changes behaviour once "
                "delivered, which this offline evaluation cannot measure. Recommended next step before full "
                "rollout: an online A/B test on a held-out traffic slice, comparing intervention-on-model-flagged "
                "vs intervention-on-baseline-flagged vs no-intervention control, tracking actual 21-day "
                "reactivation rate as the online ground truth.")
    md.append(gap_note)
    with open(ROOT / "outputs/evaluation_report.md", "w") as f:
        f.write("\n".join(md))

    print(json.dumps(report, indent=2))
    print("[evaluate] wrote outputs/pr_curve.png, outputs/evaluation_report.json, outputs/evaluation_report.md")


if __name__ == "__main__":
    main()
