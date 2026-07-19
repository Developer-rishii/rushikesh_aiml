"""
"Report precision, recall, false-positive rate on real sample data --
not 'it works'." This module is that reporting layer, computed overall
and broken down by college_tier segment so a reviewer can see whether
quality itself (not just fairness) holds up across groups.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix


def false_positive_rate(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def classification_report(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_pred)),
    }


def segmented_report(y_true, y_pred, segment) -> dict:
    y_true, y_pred, segment = np.asarray(y_true), np.asarray(y_pred), np.asarray(segment)
    out = {"overall": classification_report(y_true, y_pred)}
    for g in sorted(pd.unique(segment)):
        mask = segment == g
        if mask.sum() == 0:
            continue
        out[f"tier_{g}"] = classification_report(y_true[mask], y_pred[mask])
    return out


def disparate_impact(predictions: np.ndarray, protected: np.ndarray) -> dict:
    """(min group positive rate) / (max group positive rate) -- the 4/5ths rule."""
    protected = np.asarray(protected)
    predictions = np.asarray(predictions)
    groups = sorted(pd.unique(protected))
    rates = {}
    for g in groups:
        mask = protected == g
        n = mask.sum()
        rates[str(g)] = float(predictions[mask].mean()) if n > 0 else None

    valid_rates = [r for r in rates.values() if r is not None]
    if not valid_rates or max(valid_rates) == 0:
        di = 0.0 if valid_rates else None
    else:
        di = min(valid_rates) / max(valid_rates)
    return {"group_rates": rates, "disparate_impact": di}
