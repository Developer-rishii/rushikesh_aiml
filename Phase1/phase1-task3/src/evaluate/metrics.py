"""
evaluate module — the ONE place metrics are computed. Every stage
(train harness, tests, future notebooks) calls compute_metrics() so eval
logic can never drift or get duplicated/inconsistent between call sites.
"""
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
)

METRIC_FUNCS = {
    "pr_auc": lambda y, proba, pred: average_precision_score(y, proba),
    "roc_auc": lambda y, proba, pred: roc_auc_score(y, proba),
    "precision": lambda y, proba, pred: precision_score(y, pred),
    "recall": lambda y, proba, pred: recall_score(y, pred),
    "f1": lambda y, proba, pred: f1_score(y, pred),
    "accuracy": lambda y, proba, pred: accuracy_score(y, pred),
}


def compute_metrics(y_true, y_proba, y_pred, metric_names) -> dict:
    unknown = set(metric_names) - METRIC_FUNCS.keys()
    if unknown:
        raise ValueError(f"Unknown metric(s) requested: {unknown}. "
                          f"Available: {list(METRIC_FUNCS.keys())}")
    return {
        name: round(float(METRIC_FUNCS[name](y_true, y_proba, y_pred)), 4)
        for name in metric_names
    }
