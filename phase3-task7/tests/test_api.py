from fastapi.testclient import TestClient
from api.serve import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_version" in data

def test_api_recommend_personalized():
    res = client.get("/recommend?user_skills=python,sql&k=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["job_ids"]) == 5
    assert data["source"] == "model"
    assert "explanation" in data

def test_api_recommend_zero_skills():
    res = client.get("/recommend?user_skills=&k=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data["job_ids"]) == 5

def test_api_outage_simulation():
    # simulate outage
    res_down = client.post("/simulate_outage?down=true")
    assert res_down.json()["model_available"] is False

    # request recommendation during outage
    res_rec = client.get("/recommend?user_skills=python&k=5")
    data = res_rec.json()
    assert data["source"] == "popularity_fallback"
    assert len(data["job_ids"]) == 5

    # restore model
    client.post("/simulate_outage?down=false")

def test_api_fairness():
    res = client.get("/fairness")
    assert res.status_code == 200
    data = res.json()
    assert "demographic_parity_ratio" in data
    assert data["passed_80pct_rule"] is True
