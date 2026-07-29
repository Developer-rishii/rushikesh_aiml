"""
serve.py
========
Serving-time entry point. Deliberately separate from train_ranker.py so
this file only ever imports `compute_features` from src/features.py
(never redefines feature logic) -- that import discipline IS the
train/serve-skew fix, not a comment about one.

Failure injection support: `simulate_artifact_failure=True` corrupts the
path the registry would load from, forcing the fallback path so Stage E
can prove degraded-mode actually engages instead of just asserting it.
"""
import time

from src.baseline import SkillMatchBaseline
from src.features import feature_schema_hash


class ServingResult:
    def __init__(self, scores, model_version, degraded_mode, notes=""):
        self.scores = scores
        self.model_version = model_version
        self.degraded_mode = degraded_mode
        self.notes = notes
        self.ts = time.time()


def serve_batch(registry, batch_df, simulate_artifact_failure=False):
    prod = registry.current_production()
    if prod is None:
        raise RuntimeError("No production model has ever been promoted.")
    version, action, ts, reason = prod
    entry = registry.get(version)

    # --- train/serve schema contract check ---
    live_hash = feature_schema_hash()
    if entry["feature_schema_hash"] != live_hash:
        # Feature logic changed since this model was trained: refuse to
        # silently serve mismatched features -- fall back instead.
        fallback = SkillMatchBaseline()
        return ServingResult(
            fallback.predict(batch_df), model_version="baseline_fallback",
            degraded_mode=True,
            notes=f"feature_schema_hash mismatch (model={entry['feature_schema_hash']}, live={live_hash})",
        )

    try:
        if simulate_artifact_failure:
            raise FileNotFoundError("simulated artifact corruption/unavailability")
        model = registry.load_model(version)
        scores = model.predict(batch_df)
        return ServingResult(scores, model_version=version, degraded_mode=False)
    except Exception as e:
        fallback = SkillMatchBaseline()
        return ServingResult(
            fallback.predict(batch_df), model_version="baseline_fallback",
            degraded_mode=True, notes=f"production model unavailable: {e}",
        )
