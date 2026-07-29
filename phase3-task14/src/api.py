"""
Stage D - Per-decision explanations exposed via the API.

Run:  python3 api.py
Then: curl -X POST localhost:5000/explain -H "Content-Type: application/json" \
        -d '{"experience_years":5,"skill_match_score":80,"test_score":75,
             "applications_count":2,"college_tier":1,"pincode_tier":1}'

The model is loaded once at startup. If loading fails or MODEL_UNAVAILABLE
is toggled, /explain returns the explicit degraded response from
explainability.model_unavailable_fallback() instead of guessing.
"""
import json
import os
import joblib
from flask import Flask, request, jsonify

from paths import EXPERIMENTS_DIR
from features import MODEL_FEATURES
from explainability import explain_decision, model_unavailable_fallback

app = Flask(__name__)

MODEL_PATH = os.path.join(EXPERIMENTS_DIR, "model_mitigated.joblib")
MODEL_UNAVAILABLE = False  # flipped by the failure-injection demo
_bundle = None


def _load_model():
    global _bundle
    _bundle = joblib.load(MODEL_PATH)


try:
    _load_model()
except Exception:
    _bundle = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if _bundle and not MODEL_UNAVAILABLE else "degraded"})


@app.route("/explain", methods=["POST"])
def explain():
    global MODEL_UNAVAILABLE
    if MODEL_UNAVAILABLE or _bundle is None:
        return jsonify(model_unavailable_fallback()), 503

    payload = request.get_json(force=True)
    missing = [f for f in MODEL_FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"missing required features: {missing}"}), 400

    result = explain_decision(_bundle["clf"], _bundle["scaler"], payload)
    return jsonify(result)


@app.route("/admin/inject_failure", methods=["POST"])
def inject_failure():
    """Test-only hook used by src/failure_demo.py to prove degradation is graceful."""
    global MODEL_UNAVAILABLE
    MODEL_UNAVAILABLE = bool(request.get_json(force=True).get("unavailable", True))
    return jsonify({"MODEL_UNAVAILABLE": MODEL_UNAVAILABLE})


if __name__ == "__main__":
    app.run(port=5000)
