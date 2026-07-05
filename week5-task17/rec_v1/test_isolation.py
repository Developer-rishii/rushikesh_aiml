"""
test_isolation.py

Asserts that data isolation holds: College A cannot see College B's students.
"""

from fastapi.testclient import TestClient
from api import app
import pandas as pd
import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_tenant_isolation(client):
    """
    Test that a request from college_A trying to access a student from college_B
    is strictly rejected with a 403 Forbidden.
    """
    # 1. Find two students from different colleges
    students_df = pd.read_csv("data/students.csv")
    college_a = students_df.iloc[0]["college_id"]
    student_a = students_df.iloc[0]["student_id"]
    
    # Find a student NOT in college_a
    college_b_students = students_df[students_df["college_id"] != college_a]
    student_b = college_b_students.iloc[0]["student_id"]
    
    # 2. Attempt to query student_b using college_a's tenant context
    response = client.post(
        "/recommend",
        json={"student_id": student_b},
        headers={"x-college-id": college_a}
    )
    
    # 3. Assert it is blocked
    assert response.status_code == 403
    assert "forbidden" in response.json()["detail"].lower()
    
    # 4. Attempt to query student_a using college_a's tenant context (Should work)
    response_valid = client.post(
        "/recommend",
        json={"student_id": student_a},
        headers={"x-college-id": college_a}
    )
    assert response_valid.status_code == 200
    
def test_reverse_recommend_isolation(client):
    """
    Test that reverse recommending (finding students for a job) only ever
    returns students belonging to the requested college tenant.
    """
    jobs_df = pd.read_csv("data/jobs.csv")
    job_id = jobs_df.iloc[0]["job_id"]
    
    students_df = pd.read_csv("data/students.csv")
    college_a = students_df.iloc[0]["college_id"]
    
    response = client.post(
        "/recommend/reverse",
        json={"job_id": job_id},
        headers={"x-college-id": college_a}
    )
    assert response.status_code == 200
    
    data = response.json()
    if data["recommendations"]:
        returned_student_ids = [rec["student_id"] for rec in data["recommendations"]]
        
        # Cross check that every returned student belongs to college_a
        student_colleges = students_df[students_df["student_id"].isin(returned_student_ids)]["college_id"].unique()
        assert len(student_colleges) == 1
        assert student_colleges[0] == college_a
