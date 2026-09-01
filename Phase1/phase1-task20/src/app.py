"""
app.py
------
Flask service that loads the serialized sklearn pipeline and exposes it
behind a validated REST API.

Endpoints:
    GET  /health    -> service + model readiness
    POST /predict    -> validated single prediction
    POST /predict/batch -> validated batch prediction (realistic-scale use)

Run:
    python src/app.py
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from pydantic import ValidationError

from schemas import FEATURE_ORDER, PredictionRequest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pipeline_latest.joblib"
MODEL_VERSION = "v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("model-service")

app = Flask(__name__)

# --- Load model once at startup (fail fast, not per-request) -------------
_model = None
_model_load_error = None
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    _model_load_error = str(exc)
    logger.error("Failed to load model: %s", _model_load_error)


def _safe_errors(exc: ValidationError):
    """pydantic's exc.errors() can embed raw Exception objects (in 'ctx')
    which are not JSON serializable. Strip those down to plain strings."""
    clean = []
    for e in exc.errors(include_url=False):
        e = dict(e)
        if "ctx" in e and isinstance(e["ctx"], dict):
            e["ctx"] = {k: str(v) for k, v in e["ctx"].items()}
        clean.append(e)
    return clean


def _predict_one(features):
    start = time.perf_counter()
    X = np.array(features, dtype=float).reshape(1, -1)
    pred = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return {
        "prediction": pred,
        "label": "malignant" if pred == 0 else "benign",
        "probability_malignant": round(float(proba[0]), 6),
        "probability_benign": round(float(proba[1]), 6),
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
    }


@app.route("/health", methods=["GET"])
def health():
    """Reports service + model readiness. Used by load balancers / demos."""
    ok = _model is not None
    payload = {
        "status": "ok" if ok else "unhealthy",
        "model_loaded": ok,
        "model_version": MODEL_VERSION if ok else None,
        "feature_count_expected": len(FEATURE_ORDER),
    }
    if not ok:
        payload["error"] = _model_load_error
        return jsonify(payload), 503
    return jsonify(payload), 200


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "model_unavailable", "detail": _model_load_error}), 503

    raw = request.get_json(silent=True)
    if raw is None:
        return jsonify({"error": "bad_request", "detail": "Body must be valid JSON."}), 400

    try:
        req = PredictionRequest(**raw)
    except ValidationError as exc:
        return jsonify({"error": "validation_error", "detail": _safe_errors(exc)}), 422
    except TypeError:
        return jsonify({"error": "bad_request", "detail": "Malformed request body."}), 400

    try:
        result = _predict_one(req.features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failure")
        return jsonify({"error": "inference_error", "detail": str(exc)}), 500

    return jsonify(result), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Realistic-scale endpoint: many rows in one call, partial-failure safe."""
    if _model is None:
        return jsonify({"error": "model_unavailable", "detail": _model_load_error}), 503

    raw = request.get_json(silent=True)
    if not raw or "instances" not in raw or not isinstance(raw["instances"], list):
        return jsonify({
            "error": "bad_request",
            "detail": "Body must be {'instances': [{'features': [...]}, ...]}",
        }), 400

    if len(raw["instances"]) == 0:
        return jsonify({"error": "bad_request", "detail": "instances cannot be empty."}), 400
    if len(raw["instances"]) > 5000:
        return jsonify({
            "error": "bad_request",
            "detail": "Batch too large; max 5000 instances per call.",
        }), 400

    results, errors = [], []
    for i, item in enumerate(raw["instances"]):
        try:
            req = PredictionRequest(**item)
            results.append({"index": i, **_predict_one(req.features)})
        except ValidationError as exc:
            errors.append({"index": i, "error": "validation_error", "detail": _safe_errors(exc)})
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": i, "error": "inference_error", "detail": str(exc)})

    status = 200 if not errors else 207  # 207 Multi-Status: partial success
    return jsonify({"results": results, "errors": errors, "count": len(raw["instances"])}), status


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "not_found", "detail": "Unknown endpoint."}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "method_not_allowed", "detail": "HTTP method not supported here."}), 405


@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "internal_error", "detail": "Unexpected server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
