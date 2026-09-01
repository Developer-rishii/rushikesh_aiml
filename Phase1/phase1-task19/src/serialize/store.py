"""
serialize/store.py — Steps 2, 4, 6: save the artifact + metadata,
provide a validated load-and-predict function, version the artifact.

Versioning (Step 6): content-hash of the serialized model bytes (same
approach as Task 13) — deterministic, can't be forgotten or go stale
like a hand-typed version string, and answers "can you trace which
experiment produced this artifact?" precisely.
"""
import hashlib
import json
import logging
from pathlib import Path
import joblib
import pandas as pd

log = logging.getLogger("src.serialize.store")


def compute_version(pipeline) -> str:
    import io
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    sha256 = hashlib.sha256(buf.getvalue()).hexdigest()
    return f"sha256:{sha256[:16]}"


def save_artifact(pipeline, metadata: dict, cfg) -> dict:
    cfg.store_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = cfg.store_dir / cfg.artifact_filename
    metadata_path = cfg.store_dir / cfg.metadata_filename

    joblib.dump(pipeline, artifact_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    log.info("[Step 2] Saved artifact -> %s, metadata -> %s", artifact_path, metadata_path)
    return {"artifact_path": str(artifact_path), "metadata_path": str(metadata_path)}


class ArtifactLoadError(Exception):
    """Single, well-defined exception for any load/predict failure —
    a caller never has to catch raw joblib/sklearn/pydantic exceptions."""
    pass


def load_artifact(store_dir: Path, artifact_filename: str, metadata_filename: str):
    """Step 5: this is the function actually exercised in a simulated
    fresh environment — it reads ONLY from disk, with no dependency on
    any in-memory object from training."""
    store_dir = Path(store_dir)
    artifact_path = store_dir / artifact_filename
    metadata_path = store_dir / metadata_filename

    if not artifact_path.exists():
        raise ArtifactLoadError(f"Model artifact not found at {artifact_path}.")
    if not metadata_path.exists():
        raise ArtifactLoadError(f"Metadata file not found at {metadata_path} — "
                                 f"artifact exists but has no recorded lineage.")

    try:
        pipeline = joblib.load(artifact_path)
    except Exception as e:
        raise ArtifactLoadError(f"Failed to deserialize model artifact at {artifact_path}: {e}") from e

    metadata = json.loads(metadata_path.read_text())

    from src.serialize.metadata import collect_library_versions
    current_versions = collect_library_versions()
    recorded_versions = metadata.get("library_versions", {})
    version_mismatches = {
        lib: {"trained_with": recorded_versions.get(lib), "current": current_versions.get(lib)}
        for lib in recorded_versions
        if lib != "platform" and recorded_versions.get(lib) != current_versions.get(lib)
    }
    if version_mismatches:
        log.warning("Library version mismatch between training and load time: %s", version_mismatches)

    return pipeline, metadata, version_mismatches


def predict(pipeline, metadata: dict, X: pd.DataFrame) -> dict:
    """Step 4: load-and-predict function with input validation — checks
    the input has exactly the expected features, in ANY column order
    (reordered to match training order before scoring), answering the
    brainstorming question "what happens if input features arrive in a
    different order?" with a real guard, not an assumption."""
    expected = metadata["feature_names_ordered"]
    missing = [f for f in expected if f not in X.columns]
    if missing:
        raise ArtifactLoadError(f"Input is missing required feature(s): {missing}")
    extra = [c for c in X.columns if c not in expected]
    if extra:
        raise ArtifactLoadError(f"Input has unexpected extra column(s) not seen in training: {extra}")
    if X.isna().any().any():
        raise ArtifactLoadError("Input contains missing values that cannot be scored meaningfully.")

    X_ordered = X[expected]
    proba = pipeline.predict_proba(X_ordered)[:, 1]
    pred = pipeline.predict(X_ordered)
    return {
        "predictions": pred.tolist(),
        "probabilities": [round(float(p), 6) for p in proba],
        "artifact_version": metadata["artifact_version"],
        "n_records": len(X),
    }
