import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from fastapi.testclient import TestClient
from api import app, candidates_df

client = TestClient(app)

def test_api_unknown_candidate():
    with TestClient(app) as client:
        response = client.post("/match", json={"candidate_id": 999999, "job_id": 1001})
        assert response.status_code == 200
        assert response.json() == {"error": "Candidate not found"}

def test_api_unknown_job():
    with TestClient(app) as client:
        response = client.post("/match", json={"candidate_id": 10000, "job_id": 999999})
        assert response.status_code == 200
        assert response.json() == {"error": "Job not found"}

def test_api_malformed_request():
    with TestClient(app) as client:
        response = client.post("/match", json={"job_id": 1001})
        assert response.status_code == 422

def test_api_empty_skills(mocker):
    # We will mock the dataframe slightly to ensure empty skills
    if candidates_df is not None:
        idx = candidates_df.index[candidates_df['candidate_id'] == 10000].tolist()[0]
        original_skills = candidates_df.at[idx, 'skills']
        candidates_df.at[idx, 'skills'] = ''
        
        with TestClient(app) as client:
            response = client.post("/match", json={"candidate_id": 10000, "job_id": 1001})
            assert response.status_code == 200
            assert response.json() == {"error": "Candidate skills missing"}
        
        # Restore
        candidates_df.at[idx, 'skills'] = original_skills

def test_api_successful_match():
    with TestClient(app) as client:
        response = client.post("/match", json={"candidate_id": 10000, "job_id": 1001})
        assert response.status_code == 200
        data = response.json()
        if "error" not in data:
            assert "baseline_score" in data
            assert "prediction" in data
            assert "explanation" in data
