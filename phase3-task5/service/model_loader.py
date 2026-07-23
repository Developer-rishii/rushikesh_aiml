"""
Model loader with CWD-independent path resolution.

Design decision: resolve the model path relative to THIS file's location
(service/model_loader.py -> ../models/lgbm_ranker.txt), not the process CWD.
This prevents the silent-fallback bug where running from any directory other
than the project root would fail to find the model file.

An environment variable MODEL_PATH can override the default for deployment
flexibility (e.g., mounting a model volume in Docker).
"""
import lightgbm as lgb
import os
import json
import logging

logger = logging.getLogger(__name__)

_MODEL_INSTANCE = None
_MODEL_LOADED = False
_MODEL_METADATA = {"model_version": "1.0.0", "run_id": "unknown"}

def _default_model_path():
    """Resolve model path relative to this file, not the CWD."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", "models", "lgbm_ranker.txt"))

def _default_metadata_path():
    """Resolve model metadata path relative to this file."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", "models", "model_metadata.json"))

def get_model_metadata():
    """Return model version and MLflow run_id metadata."""
    global _MODEL_METADATA
    meta_path = os.environ.get("MODEL_METADATA_PATH", _default_metadata_path())
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                _MODEL_METADATA = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model metadata from {meta_path}: {e}")
    return _MODEL_METADATA

def load_model():
    """Load the LightGBM model. Raises on failure (never silently returns None)."""
    global _MODEL_INSTANCE, _MODEL_LOADED
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    model_path = os.environ.get("MODEL_PATH", _default_model_path())
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"CWD is {os.getcwd()}. "
            f"Set MODEL_PATH env var or ensure the model exists at the default location."
        )

    _MODEL_INSTANCE = lgb.Booster(model_file=model_path)
    _MODEL_LOADED = True
    meta = get_model_metadata()
    logger.info(f"Model loaded successfully (version: {meta.get('model_version')}, run_id: {meta.get('run_id')}).")
    return _MODEL_INSTANCE

def is_model_loaded() -> bool:
    return _MODEL_LOADED

