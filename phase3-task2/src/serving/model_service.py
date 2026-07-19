"""
The serving-side wrapper around the trained model. This is what a real
FastAPI app would call inside its /predict handler. It:
  1. Loads the LATEST versioned model artifact (or a pinned version).
  2. Computes features via the exact same feature_pipeline used in training.
  3. Times inference and records (latency, score) into the MetricsStore.
  4. Produces a plain-English explanation per prediction (top contributing
     feature by signed importance*value), for the "explainable, safe,
     demoable" bar.
  5. FAILS SAFE: if the model is unavailable/broken, falls back to the
     popularity baseline instead of throwing 500s at users, and marks
     the request as `degraded=True` so monitoring/alerting can see the
     service is running in degraded mode even though it's still "up".
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_pipeline import compute_features, FEATURE_COLUMNS, FEATURE_PIPELINE_VERSION
from src.monitoring.metrics_store import MetricsStore

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models"


class ModelUnavailableError(Exception):
    pass


class ModelService:
    def __init__(self, version=None, metrics_store: MetricsStore = None):
        self.metrics = metrics_store or MetricsStore()
        self.model = None
        self.metadata = None
        self.version = None
        self._chaos_latency_injection_ms = 0
        self._chaos_force_degenerate_output = False
        self._chaos_force_unavailable = False
        self._load(version)

    def _load(self, version=None):
        version = version or (ARTIFACT_ROOT / "LATEST_VERSION.txt").read_text().strip()
        version_dir = ARTIFACT_ROOT / version
        with open(version_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(version_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.version = version
        assert self.metadata["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION, (
            "Feature pipeline version mismatch between training artifact and serving code! "
            "This is exactly the train/serve skew this system is built to prevent -- refusing to serve."
        )

    # ---- chaos hooks used by src/chaos/inject_failure.py ----
    def chaos_inject_latency(self, ms):
        self._chaos_latency_injection_ms = ms

    def chaos_force_degenerate_output(self, on=True):
        self._chaos_force_degenerate_output = on

    def chaos_force_unavailable(self, on=True):
        self._chaos_force_unavailable = on

    def clear_chaos(self):
        self._chaos_latency_injection_ms = 0
        self._chaos_force_degenerate_output = False
        self._chaos_force_unavailable = False

    def _explain(self, features: dict):
        importances = self.metadata["feature_importances"]
        contributions = {k: importances.get(k, 0) * features[k] for k in FEATURE_COLUMNS}
        top_feat = max(contributions, key=lambda k: abs(contributions[k]))
        direction = "raised" if contributions[top_feat] >= 0 else "lowered"
        return (f"Score was mainly {direction} by '{top_feat}' "
                f"(value={features[top_feat]:.3f}, feature_importance={importances.get(top_feat,0):.3f}).")

    def predict(self, request: dict) -> dict:
        """request must contain the raw candidate/job fields (same shape
        as a training-log row). Returns score + explanation + serving
        metadata, and records the request in the metrics store."""
        t0 = time.perf_counter()
        success = True
        degraded = False
        score = None
        explanation = None
        error = None

        try:
            if self._chaos_force_unavailable:
                raise ModelUnavailableError("chaos: model process forced unavailable")

            if self._chaos_latency_injection_ms:
                time.sleep(self._chaos_latency_injection_ms / 1000.0)

            features = compute_features(request)

            if self._chaos_force_degenerate_output:
                # simulate a broken model silently returning ~constant garbage
                score = 0.5 + np.random.normal(0, 0.001)
                explanation = "DEGRADED: model output forced constant by chaos test."
                degraded = True
            else:
                import pandas as pd
                X = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                score = float(self.model.predict(X)[0])
                explanation = self._explain(features)

        except ModelUnavailableError as e:
            # ---- graceful degradation: fall back to popularity baseline ----
            success = True  # service still responds successfully to the caller
            degraded = True
            error = str(e)
            score = float(np.log1p(float(request.get("job_popularity", 1))))
            explanation = f"FALLBACK (popularity baseline) used -- primary model unavailable: {error}"
        except Exception as e:
            success = False
            degraded = True
            error = str(e)
            score = None
            explanation = f"ERROR: {error}"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.record(
            latency_ms=latency_ms, score=score, success=success,
            model_version=self.version, degraded=degraded,
            extra={"candidate_id": request.get("candidate_id"), "job_id": request.get("job_id"), "error": error},
        )

        return {
            "score": score,
            "explanation": explanation,
            "model_version": self.version,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "latency_ms": round(latency_ms, 3),
            "degraded": degraded,
            "success": success,
        }
