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
