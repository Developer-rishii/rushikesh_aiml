import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_unknown_candidate():
    response = client.post("/guardrail-check", json={"candidate_id": 9999, "job_id": 1})
    assert response.status_code == 404
    assert response.json()["error"] == "Candidate not found"

def test_unknown_job():
    response = client.post("/guardrail-check", json={"candidate_id": 1, "job_id": 9999})
    assert response.status_code == 404
    assert response.json()["error"] == "Job not found"

def test_malformed_request():
    response = client.post("/guardrail-check", json={"candidate_id": "abc"})
    assert response.status_code == 422 # Validation error

def test_successful_request():
    response = client.post("/guardrail-check", json={"candidate_id": 1, "job_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert "fit_status" in data
    assert "reason" in data
    assert "threshold_used" in data
    assert "match_score" in data
