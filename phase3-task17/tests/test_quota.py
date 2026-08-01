import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
import main as app_module
from auth import QUOTA, PLAN_LIMITS

SAMPLE_FEATURES = {"skill_overlap": 1, "seniority_gap": 1, "same_location": 0,
                    "recency_days": 10, "candidate_activity": 0.4}


def client():
    app_module.app.testing = True
    return app_module.app.test_client()


def test_rate_limit_per_minute_enforced():
    c = client()
    key = "ats-demo-key-001"
    limit = PLAN_LIMITS["standard"]["requests_per_minute"]
    statuses = []
    for i in range(limit + 3):
        r = c.post("/v1/score", headers={"X-API-Key": key},
                     json={"candidate_id": i, "features": SAMPLE_FEATURES})
        statuses.append(r.status_code)
    assert statuses.count(200) == limit, f"expected exactly {limit} successes, got {statuses.count(200)}"
    assert statuses.count(429) == 3, f"expected 3 rejections, got {statuses.count(429)}"
    print(f"PASS test_rate_limit_per_minute_enforced: {statuses.count(200)} allowed, {statuses.count(429)} throttled")


def test_abuse_pattern_broad_scraping_detected():
    c = client()
    key = "ats-gold-key-002"  # premium plan, higher per-minute ceiling so we hit
                               # the abuse heuristic, not the plain rate limit
    blocked_for_abuse = False
    for i in range(55):
        r = c.post("/v1/score", headers={"X-API-Key": key},
                     json={"candidate_id": i, "features": SAMPLE_FEATURES})  # distinct id every call
        if r.status_code == 429 and r.get_json().get("error") == "abuse_pattern_detected_broad_scraping":
            blocked_for_abuse = True
            break
    assert blocked_for_abuse, "broad, fast-scanning query pattern should trip abuse detection"
    print("PASS test_abuse_pattern_broad_scraping_detected: blocked at call", i + 1)


def test_quota_endpoint_reports_remaining():
    c = client()
    r = c.get("/v1/quota", headers={"X-API-Key": "ats-demo-key-001"})
    assert r.status_code == 200
    body = r.get_json()
    assert "requests_remaining_this_minute" in body
    print("PASS test_quota_endpoint_reports_remaining", body)


if __name__ == "__main__":
    test_rate_limit_per_minute_enforced()
    test_abuse_pattern_broad_scraping_detected()
    test_quota_endpoint_reports_remaining()
    print("\nALL QUOTA/ABUSE TESTS PASSED")
