"""
nlp/package.py — Step 6: package the text pipeline for reuse. Bundles
the winning fitted Pipeline (cleaning is a pre-step applied before the
sklearn Pipeline, documented in the config alongside it) into a single
artifact with a plain predict_category() entrypoint any caller can use
without touching internals.
"""
import json
import logging
from pathlib import Path
import joblib

log = logging.getLogger("src.nlp.package")


def package_for_reuse(fitted_pipeline, cfg, metadata: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "text_classifier_pipeline.joblib"
    joblib.dump(fitted_pipeline, model_path)

    config_path = out_dir / "pipeline_config.json"
    config_path.write_text(json.dumps({
        "cleaning": {
            "lowercase": cfg.lowercase, "strip_punctuation": cfg.strip_punctuation,
            "remove_stopwords": cfg.remove_stopwords, "min_token_length": cfg.min_token_length,
        },
        **metadata,
    }, indent=2, default=str))

    log.info("[Step 6] Packaged text pipeline for reuse -> %s", out_dir)
    return {"model_path": str(model_path), "config_path": str(config_path)}


def load_packaged_pipeline(package_dir: Path):
    package_dir = Path(package_dir)
    model_path = package_dir / "text_classifier_pipeline.joblib"
    config_path = package_dir / "pipeline_config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Incomplete package at {package_dir} — expected "
                                 f"text_classifier_pipeline.joblib and pipeline_config.json")
    model = joblib.load(model_path)
    config = json.loads(config_path.read_text())
    return model, config


def predict_category(model, cfg, raw_text: str) -> dict:
    """The one function a caller needs: raw text in, category + confidence out."""
    from src.nlp.clean_text import clean_text
    cleaned = clean_text(raw_text, cfg)
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    return {
        "predicted_category": str(classes[best_idx]),
        "confidence": round(float(proba[best_idx]), 4),
        "all_class_probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
    }
