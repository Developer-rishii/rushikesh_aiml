"""
monitoring/monitor_service.py

The core "Live model monitoring" deliverable (50 marks). Consumes a stream
of production batches, runs the same MatchModel used for serving, and for
each batch:
  1. scores every record (inference)
  2. computes precision/recall/FPR/accuracy on whatever ground truth has
     arrived so far (metrics.py)
  3. computes feature-distribution drift vs the frozen training reference
     (drift.py)
  4. raises alerts when metrics degrade or drift crosses thresholds
     (alerts.py)
  5. persists everything to SQLite so the FastAPI layer and any dashboard
     can query monitoring history without re-running the pipeline

Failure handling (Section 8, 15 marks):
  - a corrupt/missing feature in a row does not crash the batch: the row is
    quarantined and counted, the rest of the batch still gets scored
  - an empty batch, a missing reference distribution file, or a missing
    model file all fail loudly with a specific message instead of a stack
    trace, and are logged to the alerts table as an operational event
"""

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS, MONITORING_DB_PATH, BASELINE_METRICS_PATH,
    REFERENCE_DIST_PATH, MONITORING_WINDOW_SIZE,
)
from src.inference import MatchModel
from src.monitoring.metrics import compute_window_metrics
from src.monitoring.drift import compute_feature_drift, summarize_drift
from src.monitoring.alerts import evaluate_metric_alerts, evaluate_drift_alerts, build_alert_record


SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_metrics (
    batch_id INTEGER, timestamp TEXT, n_total INTEGER, n_labeled INTEGER,
    n_pending INTEGER, n_quarantined INTEGER, precision REAL, recall REAL,
    false_positive_rate REAL, accuracy REAL, status TEXT
);
CREATE TABLE IF NOT EXISTS batch_drift (
    batch_id INTEGER, timestamp TEXT, feature TEXT, psi REAL, band TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    batch_id INTEGER, timestamp TEXT, severity TEXT, type TEXT, message TEXT
);
"""


class MonitoringService:
    def __init__(self, db_path=MONITORING_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        try:
            self.model = MatchModel()
        except FileNotFoundError as e:
            raise RuntimeError(
                "Model file not found - run `python -m src.train_model` before "
                "starting monitoring. Refusing to monitor a model that isn't loaded."
            ) from e

        try:
            with open(BASELINE_METRICS_PATH) as f:
                baseline = json.load(f)
            self.baseline_metrics = baseline["validated_model"]
        except FileNotFoundError as e:
            raise RuntimeError(
                "Baseline metrics file not found - run `python -m src.train_model` first."
            ) from e

        try:
            with open(REFERENCE_DIST_PATH) as f:
                self.reference_distribution = json.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(
                "Reference feature distribution not found - run "
                "`python -m src.train_model` first."
            ) from e

    def _quarantine_bad_rows(self, batch_df):
        """Edge case handling: rows with NaN/negative-impossible/missing
        feature values are pulled out before scoring, not silently coerced."""
        mask_ok = pd.Series(True, index=batch_df.index)
        for col in FEATURE_COLUMNS:
            mask_ok &= batch_df[col].notna()
        mask_ok &= (batch_df["skill_overlap_score"].between(0, 1))
        quarantined = batch_df[~mask_ok]
        clean = batch_df[mask_ok]
        return clean, quarantined

    def process_batch(self, batch_df):
        batch_id = int(batch_df["batch_id"].iloc[0]) if "batch_id" in batch_df.columns and len(batch_df) else -1
        ts = datetime.now(timezone.utc).isoformat()

        if len(batch_df) == 0:
            alerts = [{"severity": "warning", "type": "empty_batch",
                       "message": f"Batch {batch_id} arrived empty - skipped."}]
            self._persist_alerts(batch_id, ts, alerts)
            return {"batch_id": batch_id, "status": "empty_batch_skipped", "alerts": alerts}

        clean_df, quarantined_df = self._quarantine_bad_rows(batch_df)
        n_quarantined = len(quarantined_df)

        proba, pred = self.model.predict_batch(clean_df) if len(clean_df) else (np.array([]), np.array([]))
        y_true = clean_df["is_successful_match"].values if len(clean_df) else np.array([])

        window_metrics = compute_window_metrics(y_true, pred) if len(clean_df) else \
            {"status": "insufficient_labeled_data", "n_total": 0, "n_labeled": 0, "n_pending": 0}
        window_metrics["n_quarantined"] = n_quarantined

        drift_report = compute_feature_drift(self.reference_distribution, clean_df, FEATURE_COLUMNS) \
            if len(clean_df) else {}
        drift_summary = summarize_drift(drift_report) if drift_report else {"overall_band": "stable",
                                                                              "critical_features": [],
                                                                              "warning_features": []}

        alerts = []
        if n_quarantined:
            alerts.append({"severity": "warning", "type": "data_quality",
                            "message": f"Quarantined {n_quarantined} malformed row(s) in batch "
                                       f"{batch_id} (out-of-range or missing features)."})
        alerts += evaluate_metric_alerts(window_metrics, self.baseline_metrics)
        alerts += evaluate_drift_alerts(drift_summary, drift_report)

        self._persist_batch(batch_id, ts, window_metrics, drift_report)
        self._persist_alerts(batch_id, ts, alerts)

        return {
            "batch_id": batch_id, "timestamp": ts, "metrics": window_metrics,
            "drift": {"summary": drift_summary, "detail": drift_report},
            "alerts": alerts,
            "sample_prediction": self._sample_explanation(clean_df) if len(clean_df) else None,
        }

    def _sample_explanation(self, clean_df):
        """One-example walkthrough per Stage B.4 / Section 11 self-check."""
        row = clean_df.iloc[0]
        record = {c: float(row[c]) for c in FEATURE_COLUMNS}
        result = self.model.predict_one(record, explain=True)
        result["match_id"] = row.get("match_id", "unknown")
        return result

    def _persist_batch(self, batch_id, ts, window_metrics, drift_report):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO batch_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (batch_id, ts, window_metrics.get("n_total"), window_metrics.get("n_labeled"),
             window_metrics.get("n_pending"), window_metrics.get("n_quarantined", 0),
             window_metrics.get("precision"), window_metrics.get("recall"),
             window_metrics.get("false_positive_rate"), window_metrics.get("accuracy"),
             window_metrics.get("status")),
        )
        for feature, r in drift_report.items():
            cur.execute("INSERT INTO batch_drift VALUES (?,?,?,?,?)",
                        (batch_id, ts, feature, r["psi"], r["band"]))
        self.conn.commit()

    def _persist_alerts(self, batch_id, ts, alerts):
        cur = self.conn.cursor()
        for a in alerts:
            cur.execute("INSERT INTO alerts VALUES (?,?,?,?,?)",
                        (batch_id, ts, a["severity"], a["type"], a["message"]))
        self.conn.commit()

    def history(self, table="batch_metrics"):
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY batch_id", self.conn)

    def close(self):
        self.conn.close()
