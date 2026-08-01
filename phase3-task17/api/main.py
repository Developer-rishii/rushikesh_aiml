"""
PlaceMux Public/Partner API — Task 17.

Endpoints:
  GET  /v1/health
  POST /v1/score        single candidate x job -> score + explanation
  POST /v1/match        candidate x N jobs (or job x N candidates) -> ranked list
  GET  /v1/quota        remaining quota for the caller's key

All scoring endpoints require header: X-API-Key
All scoring endpoints are versioned in the URL path (/v1/, /v2/) so an ATS
partner controls exactly when it moves to new model behaviour.
"""
from flask import Flask, request, jsonify
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from auth import authenticate, QUOTA
from versioning import load_version, ModelUnavailable, known_versions
from explain import band_score, build_explanation, unavailable_explanation

app = Flask(__name__)

FEATURES = ["skill_overlap", "seniority_gap", "same_location", "recency_days", "candidate_activity"]

# Toggle used ONLY by tests/demo to prove graceful degradation (Stage E, item 3).
_SIMULATE_OUTAGE = {"v1": False, "v2": False}


def _rule_based_fallback(feat: dict) -> float:
    """No-ML fallback used when the model is unavailable — degrades service,
    does not remove it. This is the answer to 'what happens when the model is
    unavailable' from Stage B step 4."""
    return (
        1.2 * feat.get("skill_overlap", 0)
        - 0.6 * feat.get("seniority_gap", 0)
        + 0.8 * feat.get("same_location", 0)
    )


def _require_auth():
    api_key = request.headers.get("X-API-Key")
    partner = authenticate(api_key)
    if not partner:
        return None, (jsonify(error="invalid_or_missing_api_key"), 401)
    return (api_key, partner), None


@app.route("/v1/health", methods=["GET"])
def health():
    return jsonify(status="ok", versions_available=known_versions())


@app.route("/v1/quota", methods=["GET"])
def quota():
    auth_result, err = _require_auth()
    if err:
        return err
    api_key, partner = auth_result
    return jsonify(partner=partner["name"], **QUOTA.remaining(api_key))


def _score_core(version, feat, target_id, api_key):
    ok, reason, limits = QUOTA.check_and_record(api_key, target_id=target_id)
    if not ok:
        return jsonify(
            error=reason,
            limits=limits,
            message="Request rejected by rate limiting / abuse protection.",
        ), 429

    try:
        model, meta = load_version(version, simulate_outage=_SIMULATE_OUTAGE.get(version, False))
        row = [feat.get(f, 0) for f in FEATURES]
        raw = float(model.predict([row])[0])
        reasons = build_explanation(feat, meta["feature_importances"])
        degraded = False
    except ModelUnavailable as e:
        raw = _rule_based_fallback(feat)
        reasons = unavailable_explanation()
        degraded = True
        meta = {"version": version, "model_hash": "N/A (fallback active)"}

    return jsonify(
        model_version=meta["version"],
        model_hash=meta["model_hash"],
        score_band=band_score(raw),
        reasons=reasons,
        degraded_mode=degraded,
        served_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ), 200


@app.route("/<version>/score", methods=["POST"])
def score(version):
    auth_result, err = _require_auth()
    if err:
        return err
    api_key, _ = auth_result
    body = request.get_json(force=True, silent=True) or {}
    feat = body.get("features", {})
    target_id = body.get("candidate_id")
    resp, status = _score_core(version, feat, target_id, api_key)
    return resp, status


@app.route("/<version>/match", methods=["POST"])
def match(version):
    auth_result, err = _require_auth()
    if err:
        return err
    api_key, _ = auth_result
    body = request.get_json(force=True, silent=True) or {}
    candidates = body.get("candidates", [])  # list of {candidate_id, features}
    if not candidates:
        return jsonify(error="candidates list required"), 400

    results = []
    for c in candidates:
        resp, status = _score_core(version, c.get("features", {}), c.get("candidate_id"), api_key)
        if status == 429:
            return resp, status  # quota exhausted mid-batch: fail fast, honest
        payload = resp.get_json()
        payload["candidate_id"] = c.get("candidate_id")
        results.append(payload)

    results.sort(key=lambda r: -r["score_band"])
    return jsonify(job_id=body.get("job_id"), ranked_candidates=results), 200


# --- test/demo-only control endpoint, not part of the partner contract ---
@app.route("/_admin/simulate_outage", methods=["POST"])
def simulate_outage():
    body = request.get_json(force=True, silent=True) or {}
    v = body.get("version", "v1")
    _SIMULATE_OUTAGE[v] = bool(body.get("on", False))
    return jsonify(version=v, outage_simulated=_SIMULATE_OUTAGE[v])


if __name__ == "__main__":
    app.run(port=8000)
