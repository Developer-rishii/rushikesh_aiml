"""
pipeline/artifacts.py — Step 5: save model, preprocessor, and metrics as
run artifacts. Because Step 1 chained preprocessing + model into a single
Pipeline object, "save model + preprocessor" collapses into ONE
joblib.dump() call — there's no second preprocessor artifact to
separately track, version, or accidentally let drift out of sync.
"""
import json
import logging
from pathlib import Path
import joblib

log = logging.getLogger("src.pipeline.artifacts")


def save_run_artifacts(run_dir: Path, fitted_pipeline, metrics: dict, run_metadata: dict) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = run_dir / "pipeline.joblib"
    joblib.dump(fitted_pipeline, pipeline_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2))

    log.info("Saved run artifacts -> %s (pipeline.joblib, metrics.json, run_metadata.json)", run_dir)
    return {
        "pipeline_path": str(pipeline_path),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
    }


def load_run_pipeline(run_dir: Path):
    run_dir = Path(run_dir)
    pipeline_path = run_dir / "pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"No saved pipeline artifact at {pipeline_path} — run the pipeline first.")
    return joblib.load(pipeline_path)
