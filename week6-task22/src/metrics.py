"""
src/metrics.py
Real metrics, not vibes (Study Guide Section 4).
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    out = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fpr), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y_true)),
    }
    if y_score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    return out


def best_threshold(y_true, y_score, min_precision=0.6) -> float:
    """Pick the lowest threshold that keeps precision >= min_precision,
    to maximise recall subject to a precision floor (a hiring product
    cares more about not wasting a recruiter's time on false positives)."""
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t, best_recall = 0.5, -1
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        m = compute_metrics(y_true, y_pred)
        if m["precision"] >= min_precision and m["recall"] > best_recall:
            best_t, best_recall = t, m["recall"]
    return float(best_t)


def metrics_by_segment(df, y_true_col, y_pred_col, segment_col) -> dict:
    out = {}
    for seg, sub in df.groupby(segment_col):
        out[str(seg)] = compute_metrics(sub[y_true_col], sub[y_pred_col])
    return out
