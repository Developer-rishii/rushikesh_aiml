"""
API versioning contract: a partner pins to /v1/ or /v2/. We NEVER swap the model
behind a version silently. Loading is isolated per version so a bad deploy of v2
cannot affect v1 traffic (Stage E failure-injection target).
"""
import json
import pickle
import os
from pathlib import Path

REGISTRY_DIR = str(Path(__file__).resolve().parent.parent / "ml" / "model_registry")

_CACHE = {}


class ModelUnavailable(Exception):
    pass


def load_version(version: str, simulate_outage: bool = False):
    if simulate_outage:
        raise ModelUnavailable(f"model {version} is unavailable (simulated outage)")

    if version in _CACHE:
        return _CACHE[version]

    model_path = f"{REGISTRY_DIR}/{version}/model.pkl"
    meta_path = f"{REGISTRY_DIR}/{version}/metadata.json"
    if not os.path.exists(model_path):
        raise ModelUnavailable(f"unknown or unpublished model version '{version}'")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    _CACHE[version] = (model, meta)
    return model, meta


def known_versions():
    return sorted(
        d for d in os.listdir(REGISTRY_DIR)
        if os.path.isdir(f"{REGISTRY_DIR}/{d}")
    )
