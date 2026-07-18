"""
run_pipeline.py

End-to-end run:
  1. Load 6 months of (already-generated) real-shaped data.
  2. Build features.
  3. Evaluate baseline (overlap-ratio) vs trained model on held-out data.
  4. Run the drift-monitoring + auto-retraining loop across months 4-6
     (the months with deliberate market drift baked into the data).
  5. Save model_latest.joblib for the API to serve.
  6. Write experiments/final_report.json + metrics_report.md with real numbers.

Run: python3 run_pipeline.py
"""
import os
import json
import shutil
import pandas as pd

from src.features import build_feature_matrix, FEATURE_NAMES
from src.baseline import evaluate_baseline
from src.model import MatchModel
from src.retrain_pipeline import run_monitoring_loop, MODELS_DIR
from src.metrics import metrics_by_segment

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXP_DIR = os.path.join(os.path.dirname(__file__), "experiments")
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def load_data():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    monthly = {m: pd.read_csv(os.path.join(DATA_DIR, f"interactions_{m}.csv")) for m in MONTHS}
    return students, jobs, monthly


def main():
    print("=" * 70)
    print("STAGE A -- Load data")
    print("=" * 70)
    students, jobs, monthly = load_data()
    print(f"students={len(students)} jobs={len(jobs)} months={list(monthly)}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE B.1 -- Baseline vs trained model on held-out month (2026-03)")
    print("=" * 70)
    train_months = ["2026-01", "2026-02"]
    holdout_month = "2026-03"

    train_int = pd.concat([monthly[m] for m in train_months], ignore_index=True)
    train_feats = build_feature_matrix(train_int, students, jobs)
    holdout_feats = build_feature_matrix(monthly[holdout_month], students, jobs)

    baseline_metrics = evaluate_baseline(holdout_feats)
    print(f"BASELINE (overlap-ratio only) on held-out {holdout_month}: {baseline_metrics}")

    probe_model = MatchModel(version="probe")
    probe_model.fit(train_feats)
    model_metrics = probe_model.evaluate(holdout_feats)
    print(f"TRAINED MODEL on held-out {holdout_month}:                {model_metrics}")

    lift = {
        "precision_lift": round(model_metrics["precision"] - baseline_metrics["precision"], 4),
        "recall_lift": round(model_metrics["recall"] - baseline_metrics["recall"], 4),
        "fpr_reduction": round(baseline_metrics["false_positive_rate"] - model_metrics["false_positive_rate"], 4),
    }
    print(f"LIFT over baseline: {lift}")

    # Segment breakdown by experience gap sign (a real breakdown, not one number)
    holdout_feats["_pred"] = probe_model.predict(holdout_feats)
    holdout_feats["_segment"] = holdout_feats["years_gap"].apply(
        lambda g: "underqualified" if g < 0 else "meets_or_exceeds")
    segment_report = metrics_by_segment(holdout_feats, "good_match", "_pred", "_segment")
    print(f"Segment breakdown: {json.dumps(segment_report, indent=2)}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE B.2/B.3 -- Drift monitoring + auto-retraining loop (2026-04..06)")
    print("=" * 70)
    stream_months = ["2026-04", "2026-05", "2026-06"]
    history, final_model = run_monitoring_loop(
        students, jobs, monthly,
        train_months=["2026-01", "2026-02", "2026-03"],
        stream_months=stream_months,
        min_precision=0.6,
    )

    # Promote final model as the one the API serves
    latest_src = os.path.join(MODELS_DIR, f"model_{final_model.version}.joblib")
    latest_dst = os.path.join(MODELS_DIR, "model_latest.joblib")
    shutil.copyfile(latest_src, latest_dst)
    print(f"\nPromoted {final_model.version} -> model_latest.joblib for serving.")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE C -- Verify: re-evaluate final model on ALL months (regression check)")
    print("=" * 70)
    full_report = {}
    for m in MONTHS:
        feats = build_feature_matrix(monthly[m], students, jobs)
        full_report[m] = final_model.evaluate(feats)
        print(f"  {m}: {full_report[m]}")

    # ---------------------------------------------------------------
    os.makedirs(EXP_DIR, exist_ok=True)
    report = {
        "baseline_metrics_holdout": baseline_metrics,
        "trained_model_metrics_holdout": model_metrics,
        "lift_over_baseline": lift,
        "segment_breakdown_holdout": segment_report,
        "drift_monitoring_history": history,
        "final_model_version": final_model.version,
        "final_model_all_months": full_report,
        "feature_importance_final_model": final_model.feature_importance(),
    }
    with open(os.path.join(EXP_DIR, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    write_markdown_report(report)
    print(f"\nWrote {EXP_DIR}/final_report.json and metrics_report.md")


def write_markdown_report(report):
    lines = ["# PlaceMux -- Drift + Retraining: Metrics Report\n"]
    lines.append("## Baseline vs Trained Model (held-out month 2026-03)\n")
    lines.append("| Metric | Baseline (overlap-ratio) | Trained model | Lift |")
    lines.append("|---|---|---|---|")
    b, m, l = report["baseline_metrics_holdout"], report["trained_model_metrics_holdout"], report["lift_over_baseline"]
    lines.append(f"| Precision | {b['precision']} | {m['precision']} | {l['precision_lift']:+} |")
    lines.append(f"| Recall | {b['recall']} | {m['recall']} | {l['recall_lift']:+} |")
    lines.append(f"| False Positive Rate | {b['false_positive_rate']} | {m['false_positive_rate']} | {l['fpr_reduction']:+} (reduction) |")
    lines.append(f"| ROC AUC | {b.get('roc_auc','-')} | {m.get('roc_auc','-')} | - |\n")

    lines.append("## Segment Breakdown (held-out month, trained model)\n")
    for seg, mm in report["segment_breakdown_holdout"].items():
        lines.append(f"- **{seg}** (n={mm['n']}): precision={mm['precision']}, recall={mm['recall']}, fpr={mm['false_positive_rate']}")

    lines.append("\n## Drift Monitoring & Retraining Timeline\n")
    lines.append("| Month | Drift status | Max feature PSI | Pred PSI | Event | Model | Precision | Recall | FPR |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for h in report["drift_monitoring_history"]:
        if h["event"] == "initial_train":
            mm = h["metrics"]
            lines.append(f"| {h['month']} | - | - | - | initial_train | {h['model_version']} | {mm['precision']} | {mm['recall']} | {mm['false_positive_rate']} |")
        else:
            d = h["drift"]
            mm = h["metrics_after_action"]
            lines.append(f"| {h['month']} | {d['status']} | {d['max_feature_psi']} | {d['prediction_psi']} | {h['event']} | {h['model_version']} | {mm['precision']} | {mm['recall']} | {mm['false_positive_rate']} |")

    lines.append(f"\n## Final Model: `{report['final_model_version']}`\n")
    lines.append("### Feature importance\n")
    for k, v in sorted(report["feature_importance_final_model"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")

    lines.append("\n### Regression check across all 6 months\n")
    lines.append("| Month | Precision | Recall | FPR | ROC AUC | n |")
    lines.append("|---|---|---|---|---|---|")
    for mnth, mm in report["final_model_all_months"].items():
        lines.append(f"| {mnth} | {mm['precision']} | {mm['recall']} | {mm['false_positive_rate']} | {mm.get('roc_auc','-')} | {mm['n']} |")

    with open(os.path.join(EXP_DIR, "metrics_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
