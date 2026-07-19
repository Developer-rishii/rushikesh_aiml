"""
scripts/simulate_and_monitor.py

This is the "run the whole journey once, start to finish" script
(Stage C.1) and the "2-minute live demo on real (even if small) data"
(Stage C.4). It is meant to be run and its console output + evidence/
files are the proof-of-work submitted alongside the code.

What it does, in order:
  1. Loads the production traffic sample (data/raw/production_traffic_sample.csv),
     which is time-ordered into 12 batches with drift injected from batch 7
     onward (see src/data_generator.py for exactly what is simulated and why).
  2. Feeds each batch through MonitoringService.process_batch(), which scores
     the batch with the *real* trained model, computes precision/recall/FPR
     on whatever labels have arrived, computes PSI feature drift vs the
     frozen training distribution, and raises alerts.
  3. Prints every batch's numbers and any alerts live to the console (and to
     evidence/run_logs.txt).
  4. At the end, prints one full example match end-to-end: input features,
     output score, and the plain-English "why" (Section 11 self-check /
     Section 8 "how it's checked").
  5. Saves evidence/metrics_report.json (full run) and two charts to
     evidence/plots/ showing precision/recall/FPR and PSI drift over time -
     the visual proof that monitoring caught the injected degradation.
"""

import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROD_TRAFFIC_PATH, EVIDENCE_DIR, PLOTS_DIR, MODEL_PATH, BASELINE_METRICS_PATH
from src.monitoring.monitor_service import MonitoringService
from src.utils.logging_config import get_logger

log = get_logger("simulate_and_monitor")


def ensure_trained():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(BASELINE_METRICS_PATH):
        log.info("No trained model found - training now (`python -m src.train_model`).")
        from src import train_model
        train_model.main()


def load_production_stream():
    if not os.path.exists(PROD_TRAFFIC_PATH):
        log.info("No production traffic sample found - generating simulated stream.")
        from src import data_generator
        data_generator.main()
    return pd.read_csv(PROD_TRAFFIC_PATH, parse_dates=["event_time"])


def run():
    ensure_trained()
    df = load_production_stream()
    monitor = MonitoringService()

    log.info(f"Loaded {len(df)} production events across {df['batch_id'].nunique()} time-ordered batches.")
    log.info(f"Validated baseline (from training): {monitor.baseline_metrics}")

    all_results = []
    for batch_id, batch_df in df.groupby("batch_id", sort=True):
        result = monitor.process_batch(batch_df.reset_index(drop=True))
        all_results.append(result)

        m = result["metrics"]
        drift_band = result["drift"]["summary"]["overall_band"]
        log.info(
            f"Batch {batch_id:>2} | n={m.get('n_total')} labeled={m.get('n_labeled')} "
            f"pending={m.get('n_pending')} | precision={m.get('precision')} "
            f"recall={m.get('recall')} FPR={m.get('false_positive_rate')} | "
            f"drift={drift_band}"
        )
        for alert in result["alerts"]:
            log.warning(f"  ALERT[{alert['severity']}/{alert['type']}] {alert['message']}")

    # ---- One real example, end-to-end (Section 11 self-check) -----------
    last_ok = next(r for r in reversed(all_results) if r.get("sample_prediction"))
    example = last_ok["sample_prediction"]
    log.info("=== One real example, end-to-end ===")
    log.info(f"match_id={example['match_id']} -> probability={example['match_probability']} "
             f"label={example['predicted_label']}")
    log.info(f"Why: {example['explanation']['summary']}")

    # ---- Persist full run evidence ---------------------------------------
    report_path = os.path.join(EVIDENCE_DIR, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "baseline": monitor.baseline_metrics,
            "batches": all_results,
            "example_walkthrough": example,
        }, f, indent=2, default=str)
    log.info(f"Full metrics report written to {report_path}")

    walkthrough_path = os.path.join(EVIDENCE_DIR, "demo_walkthrough.md")
    with open(walkthrough_path, "w") as f:
        f.write("# PlaceMux Live Model Monitoring - Demo Walkthrough\n\n")
        f.write("## One real example, end-to-end\n\n")
        f.write(f"- match_id: `{example['match_id']}`\n")
        f.write(f"- match probability: **{example['match_probability']}**\n")
        f.write(f"- predicted label: **{'match' if example['predicted_label'] else 'no match'}**\n")
        f.write(f"- decision threshold: {example['decision_threshold']}\n\n")
        f.write(f"**Why:** {example['explanation']['summary']}\n\n")
        f.write("| feature | value | contribution | direction |\n|---|---|---|---|\n")
        for t in example["explanation"]["top_features"]:
            f.write(f"| {t['feature']} | {t['value']} | {t['contribution']} | {t['direction']} |\n")
        f.write("\n## Batch-by-batch monitoring summary\n\n")
        f.write("| batch | n_total | n_labeled | n_pending | precision | recall | FPR | drift |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in all_results:
            m = r["metrics"]
            f.write(f"| {r['batch_id']} | {m.get('n_total')} | {m.get('n_labeled')} | "
                    f"{m.get('n_pending')} | {m.get('precision')} | {m.get('recall')} | "
                    f"{m.get('false_positive_rate')} | {r['drift']['summary']['overall_band']} |\n")
        total_alerts = sum(len(r["alerts"]) for r in all_results)
        f.write(f"\nTotal alerts raised across the run: **{total_alerts}**\n")
    log.info(f"Demo walkthrough written to {walkthrough_path}")

    _plot_metrics_over_time(all_results)
    _plot_drift_over_time(all_results)
    monitor.close()
    log.info("Run complete.")
    return all_results


def _plot_metrics_over_time(all_results):
    ok = [r for r in all_results if r["metrics"].get("status") == "ok"]
    if not ok:
        return
    batches = [r["batch_id"] for r in ok]
    precision = [r["metrics"]["precision"] for r in ok]
    recall = [r["metrics"]["recall"] for r in ok]
    fpr = [r["metrics"]["false_positive_rate"] for r in ok]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(batches, precision, marker="o", label="precision")
    ax.plot(batches, recall, marker="o", label="recall")
    ax.plot(batches, fpr, marker="o", label="false positive rate")
    ax.axvspan(7, max(batches), color="red", alpha=0.06, label="drift-injected region")
    ax.set_xlabel("production batch (time-ordered)")
    ax.set_ylabel("metric value")
    ax.set_title("Live model monitoring: precision / recall / FPR over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_over_time.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info(f"Saved {path}")


def _plot_drift_over_time(all_results):
    rows = []
    for r in all_results:
        for feat, d in r["drift"]["detail"].items():
            rows.append({"batch_id": r["batch_id"], "feature": feat, "psi": d["psi"]})
    if not rows:
        return
    dfp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for feat, g in dfp.groupby("feature"):
        ax.plot(g["batch_id"], g["psi"], marker=".", label=feat)
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="PSI warning (0.10)")
    ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="PSI critical (0.25)")
    ax.set_xlabel("production batch (time-ordered)")
    ax.set_ylabel("PSI vs training reference")
    ax.set_title("Feature drift (Population Stability Index) over time")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "drift_over_time.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info(f"Saved {path}")


if __name__ == "__main__":
    run()
