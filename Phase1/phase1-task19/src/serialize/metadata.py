"""
serialize/metadata.py — Step 3: record library versions and training
metrics. Direct guard against "no metadata/lineage": every artifact
carries enough information to answer "which experiment produced this?"
and "will this load on a clean machine with matching versions?" without
opening the pickle.
"""
import sys
import platform
import logging
from datetime import datetime, timezone

log = logging.getLogger("src.serialize.metadata")


def collect_library_versions() -> dict:
    import sklearn
    import numpy
    import pandas
    import joblib as joblib_mod
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "joblib": joblib_mod.__version__,
    }


def build_metadata(cfg, metrics: dict, features: list, split_sizes: dict, artifact_version: str) -> dict:
    metadata = {
        "artifact_version": artifact_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "model_params": cfg.model_params,
        "feature_names_ordered": features,
        "n_features": len(features),
        "target_col": cfg.target_col,
        "split_sizes": split_sizes,
        "training_metrics": metrics,
        "library_versions": collect_library_versions(),
        "source_config_path": str(cfg.config_path),
        "lineage": {
            "raw_data_path": str(cfg.raw_data_path),
            "locked_features_path": str(cfg.locked_features_path),
            "note": "Consumes Task 2's leakage-cleaned data and Task 7's locked feature set.",
        },
    }
    log.info("[Step 3] Built metadata: version=%s, %s features, metrics=%s",
              artifact_version, len(features), metrics)
    return metadata
