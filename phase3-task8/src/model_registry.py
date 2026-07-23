"""
model_registry.py
--------------------
Pitfall fixed: "No model versioning, so you cannot say which model produced
a decision six months ago."

MLflow itself needs a tracking server; we don't have one running in this
sandbox, so this is a minimal, dependency-free equivalent: every trained
model gets a content hash + a registry.json entry recording exactly which
code/data/hyperparameters produced it. `at_risk_list_for_growth.csv` and
`failure_simulation_log.json` both stamp `model_version` so any row can be
traced back to this registry entry six months from now.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def register(model_path: Path, experiment_entry: dict, registry_path: Path = None):
    registry_path = registry_path or (ROOT / "models" / "registry.json")
    with open(model_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    entry = {
        "version": experiment_entry.get("run_id", "v1"),
        "model_file": str(model_path.name),
        "content_hash_sha256_16": content_hash,
        "registered_at": pd.Timestamp.now().isoformat(),
        "training_snapshots": experiment_entry.get("train_snapshots"),
        "holdout_snapshots": experiment_entry.get("holdout_snapshots"),
        "hyperparameters": experiment_entry.get("hyperparameters"),
        "features": experiment_entry.get("features"),
        "label_horizon_days": experiment_entry.get("label_horizon_days"),
    }

    registry = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"[model_registry] registered {entry['version']} (hash={content_hash}) -> {registry_path}")
    return entry


def main():
    exp_log_path = ROOT / "experiments" / "experiment_log.jsonl"
    last_entry = [json.loads(l) for l in exp_log_path.read_text().strip().splitlines()][-1]
    model_path = ROOT / "models" / "churn_model_v1.pkl"
    register(model_path, last_entry)


if __name__ == "__main__":
    main()
