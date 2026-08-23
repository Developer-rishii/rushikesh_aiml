"""
scoring/registry.py — Step 3 (versioning half): computes a stable
version identifier for the packaged model artifact, so any score can be
traced back to exactly which model file produced it later — directly
answers "How will you know which model version scored a record later?"

The version is a content hash of model.joblib, not a hand-typed string
that could be forgotten or go stale — if the artifact file changes, the
version changes automatically and deterministically.
"""
import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger("src.scoring.registry")

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "model_artifact"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "serving_config.json"


def compute_model_version(model_path: Path = MODEL_PATH) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path} — cannot compute a version.")
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    version = f"sha256:{sha256.hexdigest()[:16]}"
    log.info("Computed model version: %s (from %s)", version, model_path)
    return version


def load_model_metadata(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Serving config not found at {config_path}.")
    return json.loads(config_path.read_text())
