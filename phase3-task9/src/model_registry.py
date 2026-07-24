"""
Pitfall being addressed: "No model versioning, so you cannot say which
model produced a decision six months ago." Every trained model is pickled
+ stamped with a version id, features, training-row count and a hash of
the training data, so any past decision is traceable back to the exact
model artifact that produced it.
"""
import json
import hashlib
import pickle
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
REGISTRY_DIR = ROOT_DIR / "reports" / "model_registry"


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:16]


def register(model, train_df: pd.DataFrame) -> dict:
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    artifact_path = os.path.join(REGISTRY_DIR, f"{model.name}_{model.version}.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "name": model.name,
        "version": model.version,
        "features": model.features,
        "trained_on_rows": len(train_df),
        "training_data_hash": _data_hash(train_df),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": artifact_path,
    }
    meta_path = os.path.join(REGISTRY_DIR, f"{model.name}_{model.version}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_registry() -> list:
    if not os.path.isdir(REGISTRY_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(REGISTRY_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(REGISTRY_DIR, fn)) as f:
                out.append(json.load(f))
    return out
