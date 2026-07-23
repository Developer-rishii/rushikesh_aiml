"""
Regression test suite for the Candidate Job Matching service.

Covers:
- Bug 1 regression: model loads regardless of CWD
- Bug 1 regression: normal prediction uses the real model, not fallback
- Failure injection works
- Edge cases: empty jobs, malformed JSON, single job, unknown candidate
- /health reflects model_loaded state
"""
import os
import sys
import json
import tempfile
import pytest

# Add service/ to path so we can import the app
SERVICE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service")
sys.path.insert(0, SERVICE_DIR)

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client. The app loads the model at startup via lifespan."""
    from app import app
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "candidate_id": "C75",
    "candidate_exp": 5,
    "candidate_skills": 4,
    "jobs": [
        {"job_id": "J12", "required_exp": 3, "required_skills": 2, "job_popularity": 0.8},
        {"job_id": "J15", "required_exp": 8, "required_skills": 6, "job_popularity": 0.2},
    ]
}


class TestModelLoader:
    """Bug 1 regression: model_loader works regardless of CWD."""

    def test_load_model_from_different_cwd(self):
        """load_model() succeeds even when CWD is a temp directory."""
        original_cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            from model_loader import load_model, _default_model_path
            # Reset the singleton so we re-load
            import model_loader
            model_loader._MODEL_INSTANCE = None
            model_loader._MODEL_LOADED = False

            path = _default_model_path()
            assert os.path.exists(path), f"Model path {path} should exist regardless of CWD={tmpdir}"

            model = load_model()
            assert model is not None
        finally:
            os.chdir(original_cwd)
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            # Reset again for other tests
            import model_loader
            model_loader._MODEL_INSTANCE = None
            model_loader._MODEL_LOADED = False


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_normal_prediction_uses_model_not_fallback(self, client):
        """Bug 1 regression: a normal request should NOT use fallback."""
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()

        assert data["used_fallback"] is False, "Normal prediction should NOT use fallback"
        assert data["fallback_reason"] is None
        assert "model_version" in data and data["model_version"] is not None
        assert "run_id" in data and data["run_id"] is not None

        # The scores should NOT equal the job_popularity values
        # (if they did, it would mean fallback is being used)
        popularity_values = {j["job_id"]: j["job_popularity"] for j in VALID_PAYLOAD["jobs"]}
        for rj in data["ranked_jobs"]:
            assert rj["score"] != popularity_values[rj["job_id"]], \
                f"Score for {rj['job_id']} equals job_popularity ({rj['score']}), " \
                f"meaning the fallback path was used instead of the real model"

    def test_failure_injection_returns_fallback(self, client):
        """x-fail-model: true should trigger fallback."""
        resp = client.post("/predict", json=VALID_PAYLOAD, headers={"x-fail-model": "true"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["used_fallback"] is True
        assert data["fallback_reason"] == "forced"

    def test_empty_jobs_list(self, client):
        """Edge case: empty jobs list should return empty ranked_jobs."""
        payload = {
            "candidate_id": "C1",
            "candidate_exp": 3,
            "candidate_skills": 2,
            "jobs": []
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ranked_jobs"] == []

    def test_single_job(self, client):
        """Edge case: single job in list."""
        payload = {
            "candidate_id": "C1",
            "candidate_exp": 3,
            "candidate_skills": 2,
            "jobs": [
                {"job_id": "J1", "required_exp": 2, "required_skills": 1, "job_popularity": 0.5}
            ]
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ranked_jobs"]) == 1
        assert data["used_fallback"] is False

    def test_malformed_json_returns_422(self, client):
        """Malformed JSON body should return 422 validation error."""
        resp = client.post("/predict", content="not valid json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422

    def test_unknown_candidate_still_works(self, client):
        """Unknown candidate_id should still produce a prediction (model only uses features)."""
        payload = {
            "candidate_id": "UNKNOWN_999",
            "candidate_exp": 10,
            "candidate_skills": 8,
            "jobs": [
                {"job_id": "J1", "required_exp": 2, "required_skills": 1, "job_popularity": 0.5}
            ]
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["used_fallback"] is False


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_when_model_loaded(self, client):
        """Health should return 200 with model_loaded: true when model is available."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True

    def test_health_reports_model_not_loaded(self):
        """If model file is missing, startup should fail (loud failure)."""
        import model_loader
        original_instance = model_loader._MODEL_INSTANCE
        original_loaded = model_loader._MODEL_LOADED

        try:
            # Reset model state
            model_loader._MODEL_INSTANCE = None
            model_loader._MODEL_LOADED = False

            # Point to a non-existent model
            os.environ["MODEL_PATH"] = "/nonexistent/model.txt"

            # Importing the app should fail at startup because model can't load
            with pytest.raises((SystemExit, Exception)):
                from app import app as test_app
                with TestClient(test_app):
                    pass
        finally:
            # Restore
            os.environ.pop("MODEL_PATH", None)
            model_loader._MODEL_INSTANCE = original_instance
            model_loader._MODEL_LOADED = original_loaded
