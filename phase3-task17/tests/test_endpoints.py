import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
import main as app_module


def client():
    app_module.app.testing = True
    return app_module.app.test_client()


SAMPLE_FEATURES = {"skill_overlap": 3, "seniority_gap": 0, "same_location": 1,
                    "recency_days": 2, "candidate_activity": 0.7}


def test_health():
    c = client()
    r = c.get("/v1/health")
    assert r.status_code == 200
    assert "v1" in r.get_json()["versions_available"]
    print("PASS test_health", r.get_json())


def test_auth_required():
    c = client()
    r = c.post("/v1/score", json={"features": SAMPLE_FEATURES})
    assert r.status_code == 401
    print("PASS test_auth_required", r.get_json())


def test_score_and_explanation_contract():
    c = client()
    r = c.post("/v1/score", headers={"X-API-Key": "ats-demo-key-001"},
                json={"candidate_id": 1, "features": SAMPLE_FEATURES})
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["score_band"] <= 100
    assert isinstance(body["reasons"], list) and len(body["reasons"]) > 0
    # contract: never leak raw feature weights or raw feature vector
    assert "feature_importances" not in str(body)
    assert body["model_version"] == "v1"
    print("PASS test_score_and_explanation_contract", body)


def test_versions_are_isolated():
    c = client()
    r1 = c.post("/v1/score", headers={"X-API-Key": "ats-demo-key-001"},
                 json={"candidate_id": 2, "features": SAMPLE_FEATURES})
    r2 = c.post("/v2/score", headers={"X-API-Key": "ats-demo-key-001"},
                 json={"candidate_id": 2, "features": SAMPLE_FEATURES})
    h1, h2 = r1.get_json()["model_hash"], r2.get_json()["model_hash"]
    assert h1 != h2, "v1 and v2 must be different models, proving version pinning matters"
    print(f"PASS test_versions_are_isolated v1_hash={h1} v2_hash={h2}")


def test_match_ranks_candidates():
    c = client()
    payload = {
        "job_id": 42,
        "candidates": [
            {"candidate_id": 1, "features": {"skill_overlap": 5, "seniority_gap": 0, "same_location": 1, "recency_days": 1, "candidate_activity": 0.9}},
            {"candidate_id": 2, "features": {"skill_overlap": 0, "seniority_gap": 3, "same_location": 0, "recency_days": 50, "candidate_activity": 0.1}},
        ],
    }
    r = c.post("/v1/match", headers={"X-API-Key": "ats-demo-key-001"}, json=payload)
    assert r.status_code == 200
    ranked = r.get_json()["ranked_candidates"]
    assert ranked[0]["candidate_id"] == 1, "stronger candidate must rank first"
    print("PASS test_match_ranks_candidates", [ (rr["candidate_id"], rr["score_band"]) for rr in ranked ])


if __name__ == "__main__":
    test_health()
    test_auth_required()
    test_score_and_explanation_contract()
    test_versions_are_isolated()
    test_match_ranks_candidates()
    print("\nALL ENDPOINT TESTS PASSED")
