"""
monitoring/drift.py

Section 12 "Go deeper": model drift detection. Implements the Population
Stability Index (PSI), the standard industry metric for comparing a live
feature/prediction distribution against a fixed training-time reference,
banded per common practice:
    PSI < 0.10            -> no significant shift
    0.10 <= PSI < 0.25     -> moderate shift, worth watching
    PSI >= 0.25            -> significant shift, investigate/retrain
"""

import numpy as np

from src.config import PSI_WARNING_THRESHOLD, PSI_CRITICAL_THRESHOLD


def psi(reference_edges, reference_proportions, live_values, epsilon=1e-4):
    edges = np.array(reference_edges)
    ref_props = np.array(reference_proportions)

    live_counts, _ = np.histogram(live_values, bins=edges)
    live_props = live_counts / max(live_counts.sum(), 1)

    ref_props = np.clip(ref_props, epsilon, None)
    live_props = np.clip(live_props, epsilon, None)

    contrib = (live_props - ref_props) * np.log(live_props / ref_props)
    return float(np.sum(contrib))


def psi_band(value):
    if value >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if value >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "stable"


def compute_feature_drift(reference_distribution, window_df, feature_columns):
    report = {}
    for col in feature_columns:
        ref = reference_distribution[col]
        value = psi(ref["edges"], ref["proportions"], window_df[col].values)
        report[col] = {"psi": round(value, 4), "band": psi_band(value)}
    return report


def summarize_drift(feature_drift_report):
    critical = [f for f, r in feature_drift_report.items() if r["band"] == "critical"]
    warning = [f for f, r in feature_drift_report.items() if r["band"] == "warning"]
    overall = "critical" if critical else ("warning" if warning else "stable")
    return {"overall_band": overall, "critical_features": critical, "warning_features": warning}
