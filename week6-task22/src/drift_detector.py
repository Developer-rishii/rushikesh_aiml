"""
src/drift_detector.py

Detects feature drift and prediction drift between a reference window
(the data the live model was trained on) and a new incoming batch,
using Population Stability Index (PSI) -- the standard, explainable
industry metric for this (no black box).

PSI interpretation (industry-standard bands):
  < 0.1            : no significant drift
  0.1 - 0.25       : moderate drift, watch
  > 0.25           : significant drift, retrain
"""
import numpy as np

PSI_RETRAIN_THRESHOLD = 0.25
PSI_WATCH_THRESHOLD = 0.10


def _psi_for_array(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_drift_report(reference_df, current_df, feature_names, model=None) -> dict:
    """Per-feature PSI + (optionally) prediction-score PSI if a model is given."""
    report = {"per_feature_psi": {}, "max_feature_psi": 0.0, "trigger_retrain": False}

    for f in feature_names:
        psi = _psi_for_array(reference_df[f].values, current_df[f].values)
        report["per_feature_psi"][f] = round(psi, 4)

    report["max_feature_psi"] = round(max(report["per_feature_psi"].values()), 4)
    report["most_drifted_feature"] = max(report["per_feature_psi"], key=report["per_feature_psi"].get)

    if model is not None:
        ref_scores = model.predict_proba(reference_df)
        cur_scores = model.predict_proba(current_df)
        report["prediction_psi"] = round(_psi_for_array(ref_scores, cur_scores), 4)
    else:
        report["prediction_psi"] = None

    drift_signals = [report["max_feature_psi"]]
    if report["prediction_psi"] is not None:
        drift_signals.append(report["prediction_psi"])

    worst = max(drift_signals)
    report["trigger_retrain"] = worst >= PSI_RETRAIN_THRESHOLD
    report["status"] = (
        "SIGNIFICANT_DRIFT" if worst >= PSI_RETRAIN_THRESHOLD else
        "MODERATE_DRIFT" if worst >= PSI_WATCH_THRESHOLD else
        "STABLE"
    )
    return report
