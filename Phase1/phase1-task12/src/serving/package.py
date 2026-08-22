"""
serving/package.py — Step 6: package the model for serving. Bundles the
fitted calibrated pipeline + chosen threshold + metadata into one
directory, and provides a single predict() entrypoint that mirrors
exactly what a real serving process would call — no separate
preprocessing step, no threshold applied inconsistently elsewhere.
"""
import json
import logging
from pathlib import Path
import joblib

log = logging.getLogger("src.serving.package")


def package_for_serving(fitted_calibrated_pipeline, threshold: float, metadata: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(fitted_calibrated_pipeline, model_path)

    config_path = out_dir / "serving_config.json"
    serving_config = {
        "threshold": threshold,
        "positive_class_meaning": "1 = benign, 0 = malignant",
        **metadata,
    }
    config_path.write_text(json.dumps(serving_config, indent=2, default=str))

    log.info("[Step 6] Packaged for serving -> %s (model.joblib + serving_config.json)", out_dir)
    return {"model_path": str(model_path), "config_path": str(config_path)}


def load_serving_package(package_dir: Path):
    package_dir = Path(package_dir)
    model_path = package_dir / "model.joblib"
    config_path = package_dir / "serving_config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Serving package incomplete at {package_dir} — "
                                 f"expected model.joblib and serving_config.json")
    model = joblib.load(model_path)
    config = json.loads(config_path.read_text())
    return model, config


def predict(model, config: dict, X):
    """The one function a serving process would actually call: raw
    features in, decision-ready {probability, decision} out — threshold
    applied exactly once, exactly here, using the packaged config."""
    proba = model.predict_proba(X)[:, 1]
    decision = (proba >= config["threshold"]).astype(int)
    return proba, decision
