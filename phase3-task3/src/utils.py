import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.config import MODEL_DIR, EXPERIMENT_LOG


def group_split(df: pd.DataFrame, group_col: str, ratios=(0.7, 0.15, 0.15), seed=42):
    """Split by job_id (query group), never by row, so no candidate list is
    partially in train and partially in test -- avoids leakage."""
    groups = df[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])
    return (
        df[df[group_col].isin(train_g)].reset_index(drop=True),
        df[df[group_col].isin(val_g)].reset_index(drop=True),
        df[df[group_col].isin(test_g)].reset_index(drop=True),
    )


def log_experiment(record: dict):
    """Append-only JSONL log so every number in the report is traceable to
    an actual run, per the 'keep the experiment log' requirement."""
    record = dict(record)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


class ModelRegistry:
    """Minimal versioned model registry (stand-in for MLflow model registry).
    Every save gets a monotonically increasing version + metadata so you can
    always answer 'which model produced this decision six months ago'."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.index_path = os.path.join(model_dir, "registry_index.json")
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w") as f:
                json.dump([], f)

    def _load_index(self):
        with open(self.index_path) as f:
            return json.load(f)

    def _save_index(self, idx):
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2)

    def save(self, name: str, model, metrics: dict, params: dict, n_bytes: int = None):
        idx = self._load_index()
        version = sum(1 for e in idx if e["name"] == name) + 1
        fname = f"{name}_v{version}.joblib"
        path = os.path.join(self.model_dir, fname)
        joblib.dump(model, path)
        size = n_bytes if n_bytes is not None else os.path.getsize(path)
        entry = {
            "name": name,
            "version": version,
            "file": fname,
            "metrics": metrics,
            "params": params,
            "size_bytes": size,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        idx.append(entry)
        self._save_index(idx)
        return entry

    def load_latest(self, name: str):
        idx = [e for e in self._load_index() if e["name"] == name]
        if not idx:
            raise KeyError(f"no model registered under {name}")
        entry = max(idx, key=lambda e: e["version"])
        model = joblib.load(os.path.join(self.model_dir, entry["file"]))
        return model, entry
