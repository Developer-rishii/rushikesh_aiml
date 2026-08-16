"""
Append-only experiment logger. Every harness run calls log_run() exactly
once — this is the single place a row gets written, so the log can't
drift out of sync with what actually ran.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path


def log_run(log_path: Path, cfg, metrics: dict, split_sizes: dict, extra: dict = None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": cfg.model_name,
        "model_params": str(cfg.model_params),
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
