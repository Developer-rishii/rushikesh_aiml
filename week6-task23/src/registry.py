"""
src/registry.py

A lightweight local Model Registry for PlaceMux.
Handles experiment tracking, model versioning, and promotion to production.
"""
import os
import json
import shutil
import datetime
from typing import Dict, Any, List

REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METADATA_FILE = os.path.join(REGISTRY_DIR, "registry_metadata.json")

class ModelRegistry:
    def __init__(self):
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        if not os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "w") as f:
                json.dump({"models": {}, "production_version": None}, f)

    def _load_metadata(self):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)

    def _save_metadata(self, data):
        with open(METADATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def log_model(self, model_obj, version: str, metrics: Dict[str, Any]):
        """Save a model and log its metrics to the registry."""
        model_path = os.path.join(REGISTRY_DIR, f"model_{version}.joblib")
        model_obj.save(model_path)
        
        metadata = self._load_metadata()
        metadata["models"][version] = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metrics": metrics,
            "path": model_path,
            "status": "Staging"
        }
        self._save_metadata(metadata)
        print(f"[Registry] Logged model {version} with precision={metrics.get('precision', '-')}")

    def promote_to_production(self, version: str):
        """Promote a specific model version to production."""
        metadata = self._load_metadata()
        if version not in metadata["models"]:
            raise ValueError(f"Model version {version} not found in registry.")
            
        metadata["production_version"] = version
        for v in metadata["models"]:
            if v == version:
                metadata["models"][v]["status"] = "Production"
            else:
                metadata["models"][v]["status"] = "Archived"
                
        self._save_metadata(metadata)
        
        # Keep the latest copy as model_latest.joblib for hardcoded failover
        src_path = metadata["models"][version]["path"]
        dst_path = os.path.join(REGISTRY_DIR, "model_latest.joblib")
        shutil.copyfile(src_path, dst_path)
        print(f"[Registry] Promoted {version} to Production.")

    def get_production_model_path(self) -> str:
        """Get the path to the current production model."""
        metadata = self._load_metadata()
        prod_version = metadata.get("production_version")
        if not prod_version:
            raise ValueError("No production model set in registry.")
        return metadata["models"][prod_version]["path"]
