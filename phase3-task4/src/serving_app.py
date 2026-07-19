"""
serving_app.py
---------------
The inference/serving path under test. This is a realistic simplification of
a production ranking service:

  * Lazy model load -> the first request pays a COLD START penalty
    (feature/model load into memory), exactly like a fresh autoscaled pod.
  * A bounded pool of "inference workers" (threading.Semaphore) models the
    real, finite compute capacity of a serving box. This is what lets a load
    test actually find a knee point instead of an artificial sleep(n).
  * When a request can't get a worker slot inside the request budget
    (QUEUE_TIMEOUT_S), the service does NOT 500. It falls back to a cheap
    heuristic ranking (skill_match + past_response_rate), tags the response
    used_fallback=true, and still returns 200 with a bounded, cheap latency.
    This is the "graceful degradation" deliverable, not a slide -- it is
    exercised for real by the failure-injection demo.
  * /admin/model_down and /admin/model_up let the failure demo turn the real
    model off entirely (e.g. bad deploy, OOM) and confirm the service still
    serves via fallback rather than dying.
"""
import json
import os
import threading
import time

import joblib
import numpy as np
from flask import Flask, jsonify, request

SERVED_FEATURES_LOG = "results/served_features.jsonl"
_log_lock = threading.Lock()


def _log_served_features(candidates, job_features, feats):
    # Sampled, append-only log of exactly what feature values were used at
    # serving time -- this is what a real train/serve-skew monitor diffs
    # against the training-time feature stats (see src/skew_check.py).
    try:
        with _log_lock, open(SERVED_FEATURES_LOG, "a") as f:
            for c in candidates[:3]:  # sample a few per request, not the whole batch
                row = {feat: {**c, **job_features}.get(feat, 0) for feat in feats}
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass

app = Flask(__name__)

MODEL_PATH = "models/ranker_model.joblib"
N_INFERENCE_WORKERS = 8          # capacity of this instance -- the thing we are load-testing
QUEUE_TIMEOUT_S = 0.35           # how long a request will wait for a free worker before falling back
BASE_INFERENCE_MS = 12           # fixed per-request model compute cost
PER_CANDIDATE_MS = 0.6           # scales with list size, like a real feature+scoring pass
COLD_START_PENALTY_S = 0.9       # one-time model/feature load cost

_worker_slots = threading.Semaphore(N_INFERENCE_WORKERS)
_state_lock = threading.Lock()
_state = {
    "model": None,
    "features": None,
    "loaded": False,
    "model_forced_down": False,   # flipped by /admin/model_down for the failure demo
    "request_count": 0,
    "fallback_count": 0,
    "cold_start_events": 0,
}


def _lazy_load():
    with _state_lock:
        if not _state["loaded"]:
            time.sleep(COLD_START_PENALTY_S)  # simulate model + feature-store load
            bundle = joblib.load(MODEL_PATH)
            _state["model"] = bundle["model"]
            _state["features"] = bundle["features"]
            _state["loaded"] = True
            _state["cold_start_events"] += 1


def _heuristic_fallback_score(cands):
    # Cheap, dependency-free ranking: no model call, no feature store round trip.
    return [0.7 * c.get("skill_match", 0) + 0.3 * c.get("past_response_rate", 0) for c in cands]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _state["loaded"],
        "model_forced_down": _state["model_forced_down"],
        "request_count": _state["request_count"],
        "fallback_count": _state["fallback_count"],
        "cold_start_events": _state["cold_start_events"],
    })


@app.route("/admin/model_down", methods=["POST"])
def model_down():
    _state["model_forced_down"] = True
    return jsonify({"model_forced_down": True})


@app.route("/admin/model_up", methods=["POST"])
def model_up():
    _state["model_forced_down"] = False
    return jsonify({"model_forced_down": False})


@app.route("/rank", methods=["POST"])
def rank():
    t_start = time.time()
    with _state_lock:
        _state["request_count"] += 1
    payload = request.get_json(force=True)
    candidates = payload.get("candidates", [])
    job_features = payload.get("job", {})

    if _state["model_forced_down"]:
        scores = _heuristic_fallback_score(candidates)
        with _state_lock:
            _state["fallback_count"] += 1
        latency_ms = (time.time() - t_start) * 1000
        return jsonify(_response(candidates, scores, True, "model_forced_down", latency_ms))

    if not _state["loaded"]:
        _lazy_load()

    acquired = _worker_slots.acquire(timeout=QUEUE_TIMEOUT_S)
    if not acquired:
        # Every inference worker is busy beyond our request budget -> degrade, don't fail.
        scores = _heuristic_fallback_score(candidates)
        with _state_lock:
            _state["fallback_count"] += 1
        latency_ms = (time.time() - t_start) * 1000
        return jsonify(_response(candidates, scores, True, "capacity_exceeded", latency_ms))

    try:
        # simulate real inference compute cost, proportional to list size
        compute_s = (BASE_INFERENCE_MS + PER_CANDIDATE_MS * len(candidates)) / 1000.0
        time.sleep(compute_s)
        model, feats = _state["model"], _state["features"]
        rows = np.array([[{**c, **job_features}.get(f, 0) for f in feats] for c in candidates])
        scores = model.predict(rows).tolist()
        _log_served_features(candidates, job_features, feats)
    finally:
        _worker_slots.release()

    latency_ms = (time.time() - t_start) * 1000
    return jsonify(_response(candidates, scores, False, None, latency_ms))


def _response(candidates, scores, used_fallback, fallback_reason, latency_ms):
    order = np.argsort(-np.array(scores))
    ranked = [candidates[i].get("candidate_id", i) for i in order]
    return {
        "ranked_candidate_ids": ranked,
        "scores": scores,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "server_latency_ms": round(latency_ms, 2),
    }


if __name__ == "__main__":
    # threaded=True: this Flask dev server handles concurrent requests on
    # real OS threads, so semaphore-bound capacity behaves like a real box.
    app.run(host="127.0.0.1", port=8877, threaded=True)
