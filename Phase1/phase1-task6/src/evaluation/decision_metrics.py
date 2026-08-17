"""
evaluation/decision_metrics.py — Step 2/3: confusion matrix, precision,
recall, F1 at a given decision threshold. Everything here operates on
VALIDATION probabilities, never training data.

Clinical label semantics for this dataset (documented once, used
everywhere): target=0 -> malignant, target=1 -> benign. sklearn treats
class 1 (benign) as "positive" by convention, which means:
  - sklearn's "false positive" (true=0, pred=1) = a MISSED MALIGNANCY
    (predicted benign, actually malignant) -- the dangerous error.
  - sklearn's "false negative" (true=1, pred=0) = an UNNECESSARY BIOPSY
    (predicted malignant, actually benign) -- the low-cost error.
This module names both explicitly so the cost reasoning in
threshold_selection.py can never be misapplied to the wrong cell.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
)


def confusion_matrix_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, missed_malignancy, unnecessary_biopsy, tp = cm.ravel()
    # tn = true=0,pred=0 (correctly caught malignant)
    # missed_malignancy = true=0,pred=1 (sklearn's FP; dangerous)
    # unnecessary_biopsy = true=1,pred=0 (sklearn's FN; low-cost)
    # tp = true=1,pred=1 (correctly cleared benign)
    return {
        "threshold": round(float(threshold), 4),
        "true_negative_malignant_caught": int(tn),
        "missed_malignancy": int(missed_malignancy),
        "unnecessary_biopsy": int(unnecessary_biopsy),
        "true_positive_benign_cleared": int(tp),
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_labels": ["true=0 (malignant)", "true=1 (benign)"],
    }


def decision_metrics_at_threshold(y_true, y_proba, threshold: float, metric_names) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    funcs = {
        "precision": lambda: precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "roc_auc": lambda: roc_auc_score(y_true, y_proba),  # threshold-independent
        "pr_auc": lambda: average_precision_score(y_true, y_proba),  # threshold-independent
    }
    unknown = set(metric_names) - funcs.keys()
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {list(funcs.keys())}")
    return {name: round(float(funcs[name]()), 4) for name in metric_names}
