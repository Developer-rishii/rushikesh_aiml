"""
pipeline/evaluate.py — Step 3: attach evaluation and metric logging at
the end of the pipeline. Single source of metric computation, called
with validation data only.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
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
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {list(METRIC_FUNCS.keys())}")
    return {name: round(float(METRIC_FUNCS[name](y_true, y_proba, y_pred)), 4) for name in metric_names}


def evaluate_pipeline(fitted_pipeline, X_val, y_val, metric_names) -> dict:
    y_proba = fitted_pipeline.predict_proba(X_val)[:, 1]
    y_pred = fitted_pipeline.predict(X_val)
    return compute_metrics(y_val, y_proba, y_pred, metric_names)


def log_run(log_path: Path, run_id: str, cfg, metrics: dict, split_sizes: dict, extra: dict = None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "train_size": split_sizes["train"],
        "val_size": split_sizes["val"],
        "test_size": split_sizes["test"],
        "primary_metric": cfg.primary_metric,
    }
    row.update({f"val_{k}": v for k, v in metrics.items()})
    if extra:
        row.update(extra)

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row
