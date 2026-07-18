"""
src/baseline.py

The dumb baseline required by Study Guide Section 4 ("Baseline first"):
rank purely by overlap of required vs verified skills. Every later
number (the trained model) is only meaningful relative to this.
"""
from .metrics import compute_metrics, best_threshold


def baseline_score(feature_df):
    """Score = overlap_ratio only. No weighting, no learning."""
    return feature_df["overlap_ratio"].values


def evaluate_baseline(feature_df, min_precision=0.6):
    y_true = feature_df["good_match"].values
    y_score = baseline_score(feature_df)
    t = best_threshold(y_true, y_score, min_precision=min_precision)
    y_pred = (y_score >= t).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["threshold"] = t
    return metrics
