"""
Stage E step 3: deliberately induce the failure and confirm the designed
degradation actually happens (never a bare 500 to a partner).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
import main as app_module

SAMPLE_FEATURES = {"skill_overlap": 4, "seniority_gap": 0, "same_location": 1,
                    "recency_days": 3, "candidate_activity": 0.8}


def client():
    app_module.app.testing = True
    return app_module.app.test_client()


def test_model_outage_degrades_gracefully_not_a_500():
    c = client()
    key = "ats-demo-key-001"

    # 1. confirm healthy baseline behaviour first
    r_before = c.post("/v1/score", headers={"X-API-Key": key},
                        json={"candidate_id": 900, "features": SAMPLE_FEATURES})
    assert r_before.status_code == 200
    assert r_before.get_json()["degraded_mode"] is False

    # 2. induce the failure
    c.post("/_admin/simulate_outage", json={"version": "v1", "on": True})

    # 3. confirm partner still gets a usable, explained response, not a crash
    r_during = c.post("/v1/score", headers={"X-API-Key": key},
                        json={"candidate_id": 901, "features": SAMPLE_FEATURES})
    assert r_during.status_code == 200, "outage must degrade, never hard-fail the partner"
    body = r_during.get_json()
    assert body["degraded_mode"] is True
    assert "fallback" in body["reasons"][0].lower()
    assert 0 <= body["score_band"] <= 100

    # 4. recovery
    c.post("/_admin/simulate_outage", json={"version": "v1", "on": False})
    r_after = c.post("/v1/score", headers={"X-API-Key": key},
                       json={"candidate_id": 902, "features": SAMPLE_FEATURES})
    assert r_after.get_json()["degraded_mode"] is False

    print("PASS test_model_outage_degrades_gracefully_not_a_500")
    print("  before:", r_before.get_json())
    print("  during outage:", body)
    print("  after recovery:", r_after.get_json())


if __name__ == "__main__":
    test_model_outage_degrades_gracefully_not_a_500()
    print("\nALL FAILURE-MODE TESTS PASSED")
