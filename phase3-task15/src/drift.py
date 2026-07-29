"""
drift.py
========
Drift detection with retraining triggers and a rollback path (Stage C).

Two independent signals are monitored, because covariate drift and concept
drift can each happen without the other (Section 4 definition: "input OR
outcome distributions shifting"):

  1. INPUT (covariate) drift: Population Stability Index (PSI) per feature,
     reference = training distribution, comparison = latest served batch.
     PSI > 0.25 on any feature => significant drift (industry-standard cut,
     0.1-0.25 = moderate/watch, <0.1 = stable).

  2. OUTCOME/performance drift: once labels for a batch are available
     (with the natural application/shortlist delay), offline nDCG@10 on that
     batch is compared to the metric the production model was registered
     with. A drop beyond `perf_drop_threshold` triggers retraining even if
     PSI looks fine (covers concept drift where inputs look the same but
     the input->label relationship changed, exactly what generate_data.py
     injects after day 120).

DESIGN DECISION (Section 8, written down as required):
  Retraining trigger = DRIFT-TRIGGERED, not purely scheduled.
  Rejected pure scheduled retraining because PlaceMux's drift can arrive in
  a single day (e.g. an embedding-model swap) and a weekly/monthly cadence
  would leave a stale, silently-wrong model serving for days. We keep a
  scheduled MINIMUM cadence too (see run_end_to_end.py) as a safety net in
  case drift signals themselves fail silently -- belt and suspenders.

  Promotion of a retrained candidate = HUMAN-IN-THE-LOOP approval gate, not
  fully automatic. Rejected full automation because this model affects
  hiring outcomes (DPDP + fairness constraints, Section 3); an automatic
  promotion could silently ship a fairness regression. The registry's
  `promote()` always logs an `approved_by`; in this repo the pipeline
  approves candidates that pass BOTH the offline-beats-baseline gate AND
  the fairness gate, and the audit trail records exactly that.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for one numeric feature."""
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=cut_points)
    comp_hist, _ = np.histogram(comparison, bins=cut_points)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-4, None)
    comp_pct = np.clip(comp_hist / max(comp_hist.sum(), 1), 1e-4, None)
    return float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))


class DriftDetector:
    def __init__(self, feature_columns, psi_warn=0.10, psi_alert=0.25, perf_drop_threshold=0.08,
                 report_dir="governance/drift_reports"):
        self.feature_columns = feature_columns
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.perf_drop_threshold = perf_drop_threshold
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def check_input_drift(self, reference_df, comparison_df) -> dict:
        scores = {}
        for col in self.feature_columns:
            scores[col] = round(psi(reference_df[col].values, comparison_df[col].values), 4)
        max_feat = max(scores, key=scores.get)
        status = "ALERT" if scores[max_feat] >= self.psi_alert else (
            "WARN" if scores[max_feat] >= self.psi_warn else "STABLE")
        return {"per_feature_psi": scores, "max_psi_feature": max_feat,
                "max_psi": scores[max_feat], "status": status}

    def check_performance_drift(self, baseline_metric: float, current_metric: float) -> dict:
        drop = baseline_metric - current_metric
        rel_drop = drop / baseline_metric if baseline_metric else 0.0
        triggered = rel_drop >= self.perf_drop_threshold
        return {"baseline_metric": baseline_metric, "current_metric": current_metric,
                "relative_drop": round(rel_drop, 4), "retrain_triggered": triggered}

    def evaluate_and_log(self, reference_df, comparison_df, baseline_metric, current_metric, batch_name):
        input_report = self.check_input_drift(reference_df, comparison_df)
        perf_report = self.check_performance_drift(baseline_metric, current_metric)
        retrain = perf_report["retrain_triggered"] or input_report["status"] == "ALERT"
        report = {
            "batch": batch_name,
            "ts": time.time(),
            "input_drift": input_report,
            "performance_drift": perf_report,
            "retrain_triggered": retrain,
            "trigger_reason": (
                "performance drop" if perf_report["retrain_triggered"] else
                "input distribution alert" if input_report["status"] == "ALERT" else
                "none"
            ),
        }
        out_path = self.report_dir / f"drift_report_{batch_name}.json"
        out_path.write_text(json.dumps(report, indent=2))
        return report
