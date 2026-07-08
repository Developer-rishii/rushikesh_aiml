import pytest
from fastapi.testclient import TestClient
from src.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_unknown_college(client):
    """Test accessing an unknown college returns 404 gracefully."""
    response = client.get("/college/college_UNKNOWN/recommendations")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_empty_college(client):
    """Test accessing a college with zero data returns 404 gracefully."""
    response = client.get("/college/college_EMPTY/recommendations")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_unknown_student(client):
    """Test accessing an unknown student returns 404 gracefully."""
    response = client.get("/college/college_A/student/UNKNOWN_STUDENT/job/job_0")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_unknown_job(client):
    """Test accessing an unknown job for a valid student returns 404 gracefully."""
    response = client.get("/college/college_A/student/fresh_college_A_0/job/UNKNOWN_JOB")
    assert response.status_code == 404
    assert "found" in response.json()["detail"].lower()

def test_malformed_request_is_handled(client):
    """Test hitting an undefined endpoint returns 404 gracefully."""
    response = client.get("/college/invalid/path/format")
    assert response.status_code == 404


