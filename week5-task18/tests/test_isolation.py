import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


def test_cross_college_isolation():
    """Cross-college isolation: one college cannot see another college's data."""
    from fastapi.testclient import TestClient
    from main import app, df

    if df.empty:
        pytest.skip("Processed data not loaded — run pipeline first")

    client = TestClient(app)

    # Pick a real row
    row = df.iloc[0]
    college_id = row['college_id']
    student_id = row['student_id']
    job_id = row['job_id']

    # Valid call — same college
    resp = client.get(f"/explain/{college_id}/{student_id}/{job_id}")
    assert resp.status_code == 200, "Should return data for correct college"

    # Invalid call — wrong college should be 404
    wrong_college = "C999" if college_id != "C999" else "C888"
    resp_wrong = client.get(f"/explain/{wrong_college}/{student_id}/{job_id}")
    assert resp_wrong.status_code == 404, \
        f"College {wrong_college} should NOT see student {student_id}'s data"

    # Dashboard isolation: wrong college returns 404
    resp_dash = client.get(f"/portal/{wrong_college}/dashboard")
    assert resp_dash.status_code == 404, \
        f"Dashboard for non-existent college {wrong_college} should be 404"
