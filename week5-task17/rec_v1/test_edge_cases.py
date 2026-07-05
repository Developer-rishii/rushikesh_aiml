"""
test_edge_cases.py

Validates API handling of edge cases and bad inputs.
"""

from fastapi.testclient import TestClient
from api import app
import pandas as pd
import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
        
@pytest.fixture(scope="module")
def students_df():
    return pd.read_csv("data/students.csv")

def test_unknown_student(client, students_df):
    """Test querying a student ID that does not exist returns 404."""
    college_id = students_df.iloc[0]["college_id"]
    response = client.post(
        "/recommend",
        json={"student_id": "student_9999999"},
        headers={"x-college-id": college_id}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    
def test_malformed_request(client, students_df):
    """Test missing required body fields returns 422 Unprocessable Entity."""
    college_id = students_df.iloc[0]["college_id"]
    response = client.post(
        "/recommend",
        json={"wrong_field": "student_0"},
        headers={"x-college-id": college_id}
    )
    assert response.status_code == 422
    
def test_cold_start_student(client, students_df):
    """Test a student with zero verified skills gets handled gracefully."""
    # We injected 50 cold start students in generate_data.py
    # Let's find one by looking at student_skills.csv
    student_skills_df = pd.read_csv("data/student_skills.csv")
    all_students = set(students_df["student_id"])
    skilled_students = set(student_skills_df["student_id"])
    cold_start_students = list(all_students - skilled_students)
    
    if not cold_start_students:
        pytest.skip("No cold start students found in dataset.")
        
    cold_student = cold_start_students[0]
    college_id = students_df[students_df["student_id"] == cold_student].iloc[0]["college_id"]
    
    response = client.post(
        "/recommend",
        json={"student_id": cold_student},
        headers={"x-college-id": college_id}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 0
    assert "Cold start" in data["message"]
    
def test_unknown_job_reverse_recommend(client, students_df):
    college_id = students_df.iloc[0]["college_id"]
    response = client.post(
        "/recommend/reverse",
        json={"job_id": "job_999999"},
        headers={"x-college-id": college_id}
    )
    assert response.status_code == 404

def test_explanation_consistency(client, students_df):
    """Test that if a worse-featured job outranks a better-featured job, the explanation explicitly accounts for it."""
    valid_student = students_df.iloc[0]["student_id"]
    college_id = students_df.iloc[0]["college_id"]
    
    response = client.post(
        "/recommend",
        json={"student_id": valid_student},
        headers={"x-college-id": college_id}
    )
    assert response.status_code == 200
    recs = response.json().get("recommendations", [])
    if len(recs) < 2:
        pytest.skip("Not enough recommendations to test consistency.")
        
    from features import FeatureEngineer
    import joblib
    fe = FeatureEngineer()
    fe.load_context()
    artifact = joblib.load("model.pkl")
    fe.priors = artifact["college_priors"]
    
    job_ids = [r["job_id"] for r in recs]
    pairs = pd.DataFrame({"student_id": [valid_student]*len(job_ids), "job_id": job_ids})
    features_df = fe.transform(pairs)
    features_df["job_id"] = job_ids
    
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            rec_high = recs[i]
            rec_low = recs[j]
            feat_high = features_df[features_df["job_id"] == rec_high["job_id"]].iloc[0]
            feat_low = features_df[features_df["job_id"] == rec_low["job_id"]].iloc[0]
            
            worse_skill = feat_high["skill_overlap_ratio"] < feat_low["skill_overlap_ratio"]
            worse_gap = feat_high["proficiency_gap"] > feat_low["proficiency_gap"]
            worse_exp = feat_high["experience_fit"] < feat_low["experience_fit"]
            
            if worse_skill and worse_gap and worse_exp:
                # If strictly worse but outranks, it MUST explicitly say why (e.g. college prior)
                expl = rec_high["explanation"].lower()
                assert "strong historical placement outcomes" in expl or "above-average hire rate" in expl, \
                    f"Contradiction found! {rec_high['job_id']} outranks {rec_low['job_id']} despite worse core features, but explanation does not explicitly state the college prior pull-up. Explanation: {expl}"
