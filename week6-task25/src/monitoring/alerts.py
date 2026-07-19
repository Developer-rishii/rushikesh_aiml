"""
monitoring/alerts.py

Turns metric/drift numbers into concrete, actionable alerts. This is what
makes the difference between "we log numbers nobody reads" and "monitoring"
- Section 9's pitfall: "We can tighten security/monitoring after launch is
too late" only holds if alerts actually fire and are visible before launch.
"""

from datetime import datetime, timezone

from src.config import METRIC_DEGRADATION_ALERT_THRESHOLD
from src.monitoring.metrics import relative_change


def evaluate_metric_alerts(window_metrics, baseline_metrics):
    """Compare a window's precision/recall/FPR against the validated
    baseline run from training. Precision/recall dropping, or FPR rising,
    beyond the configured relative threshold raises an alert."""
    alerts = []
    if window_metrics["status"] != "ok":
        alerts.append({
            "severity": "info",
            "type": "insufficient_data",
            "message": (f"Only {window_metrics['n_labeled']} labeled outcomes in this "
                        f"window ({window_metrics['n_pending']} pending) - below the "
                        f"minimum trusted sample size. Metrics withheld, not faked."),
        })
        return alerts

    checks = [
        ("precision", window_metrics["precision"], baseline_metrics["precision"], "drop_is_bad"),
        ("recall", window_metrics["recall"], baseline_metrics["recall"], "drop_is_bad"),
        ("false_positive_rate", window_metrics["false_positive_rate"],
         baseline_metrics["false_positive_rate"], "rise_is_bad"),
    ]
    for name, current, baseline, direction in checks:
        change = relative_change(current, baseline)
        if change is None:
            continue
        bad = (direction == "drop_is_bad" and change <= -METRIC_DEGRADATION_ALERT_THRESHOLD) or \
              (direction == "rise_is_bad" and change >= METRIC_DEGRADATION_ALERT_THRESHOLD)
        if bad:
            alerts.append({
                "severity": "critical",
                "type": "metric_degradation",
                "metric": name,
                "baseline": baseline,
                "current": current,
                "relative_change": round(change, 4),
                "message": (f"Live {name.replace('_', ' ')} is {current:.3f} vs validated "
                            f"baseline {baseline:.3f} ({change:+.1%}) - exceeds the "
                            f"{METRIC_DEGRADATION_ALERT_THRESHOLD:.0%} degradation threshold."),
            })
    return alerts


def evaluate_drift_alerts(drift_summary, feature_drift_report):
    alerts = []
    if drift_summary["overall_band"] == "critical":
        alerts.append({
            "severity": "critical",
            "type": "feature_drift",
            "features": drift_summary["critical_features"],
            "message": (f"Critical distribution shift (PSI >= 0.25) on: "
                        f"{', '.join(drift_summary['critical_features'])}. "
                        f"Investigate upstream data source before trusting live scores."),
        })
    elif drift_summary["overall_band"] == "warning":
        alerts.append({
            "severity": "warning",
            "type": "feature_drift",
            "features": drift_summary["warning_features"],
            "message": (f"Moderate distribution shift (0.10<=PSI<0.25) on: "
                        f"{', '.join(drift_summary['warning_features'])}. Watch closely."),
        })
    return alerts


def build_alert_record(batch_id, alerts):
    return {
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert_count": len(alerts),
        "alerts": alerts,
    }
