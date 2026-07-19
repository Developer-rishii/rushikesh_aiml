"""
monitoring/metrics.py

Section 4: "Real metrics, not vibes ... Report precision, recall,
false-positive rate on real sample data ... A guardrail metric protects the
product from silent degradation."

This module computes those numbers per monitoring window, and is explicit
about the realistic constraint that not every live prediction has a
ground-truth label yet (hiring outcomes lag by days/weeks). Windows with too
few labeled samples are flagged "insufficient_data" instead of reporting a
misleadingly precise number.
"""

from src.config import MIN_LABELED_SAMPLES_FOR_METRICS


def compute_window_metrics(y_true, y_pred):
    """y_true may contain -1 for 'outcome not yet known' - those rows are
    excluded from metric computation but the exclusion count is reported."""
    labeled_mask = (y_true != -1)
    n_total = len(y_true)
    n_labeled = int(labeled_mask.sum())
    n_pending = n_total - n_labeled

    if n_labeled < MIN_LABELED_SAMPLES_FOR_METRICS:
        return {
            "status": "insufficient_labeled_data",
            "n_total": n_total, "n_labeled": n_labeled, "n_pending": n_pending,
            "precision": None, "recall": None, "false_positive_rate": None,
            "accuracy": None,
        }

    yt = y_true[labeled_mask]
    yp = y_pred[labeled_mask]

    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / n_labeled if n_labeled else 0.0

    return {
        "status": "ok",
        "n_total": n_total, "n_labeled": n_labeled, "n_pending": n_pending,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4), "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def relative_change(current, baseline):
    if baseline in (None, 0):
        return None
    return (current - baseline) / baseline
