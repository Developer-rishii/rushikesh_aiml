"""
Rec v1 — pytest tests for edge cases, data isolation, and dependency validation.

Every test is named explicitly and referenced in the sign-off report.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app, load_data

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def _startup():
    """Load data once for all tests in this module."""
    load_data()
    yield


# ── TEST 1: Cross-college data isolation (highest-priority) ─────────────────

class TestCrossCollegeIsolation:
    """
    Proves that college A cannot access college B's student data.
    Referenced in sign-off report § Data Isolation Proof.
    """

    def test_cross_college_isolation(self):
        """
        Request recs for student_B_0 (belongs to college_B) via college_A's endpoint.
        Must return 403, NOT that student's data.
        """
        response = client.get("/recommend/college_A/student_B_0")
        assert response.status_code == 403, (
            f"Expected 403 for cross-college access, got {response.status_code}: "
            f"{response.json()}"
        )
        assert "Forbidden" in response.json()["detail"]
        assert "Cross-college access denied" in response.json()["detail"]

    def test_college_B_students_not_in_college_A_dashboard(self):
        """
        College A's dashboard must not report college B's students.
        """
        resp_a = client.get("/portal/college_A/dashboard")
        assert resp_a.status_code == 200
        dash_a = resp_a.json()

        resp_b = client.get("/portal/college_B/dashboard")
        assert resp_b.status_code == 200
        dash_b = resp_b.json()

        # College A and B must have different student counts
        assert dash_a["metrics"]["enrolled_students"] != dash_b["metrics"]["enrolled_students"], (
            "College A and B report the same student count — possible data leak"
        )

    def test_reverse_cross_college_isolation(self):
        """
        Also test the reverse: college_B cannot access college_A's student.
        """
        response = client.get("/recommend/college_B/student_A_0")
        assert response.status_code == 403

    def test_cross_college_explain_blocked(self):
        """
        The /explain endpoint must also enforce isolation.
        """
        response = client.get("/recommend/college_A/student_B_0/explain")
        assert response.status_code == 403


# ── TEST 2: Matching v1 schema validation ────────────────────────────────────

class TestMatchingV1SchemaValidation:
    """
    Validates that the pipeline fails loudly when matching_v1_output.csv is
    malformed (specifically: missing college_id).
    Referenced in sign-off report § Upstream Dependency.
    """

    def test_schema_rejects_missing_college_id(self):
        """Feed a malformed CSV missing college_id and confirm failure."""
        from src.ranking import validate_matching_schema

        good_df = pd.read_csv("data/matching_v1_output.csv")
        bad_df = good_df.drop(columns=["college_id"])

        with pytest.raises(ValueError, match="Missing columns.*college_id"):
            validate_matching_schema(bad_df)

    def test_schema_accepts_valid_file(self):
        """The real file must pass validation without error."""
        from src.ranking import validate_matching_schema

        df = pd.read_csv("data/matching_v1_output.csv")
        validate_matching_schema(df)  # should not raise


# ── TEST 3: Zero-candidate student ───────────────────────────────────────────

class TestZeroCandidateStudent:
    """
    student_B_zero belongs to college_B but has zero rows in matching_v1_output.csv.
    Must return an empty recommendation list with a clear reason — not an error.
    Referenced in sign-off report § Edge Cases.
    """

    def test_zero_candidate_returns_empty_list(self):
        response = client.get("/recommend/college_B/student_B_zero")
        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"] == []
        assert "No candidate jobs found" in data["reason"]


# ── TEST 4: Single-student college ───────────────────────────────────────────

class TestSingleStudentCollege:
    """
    college_D has only 2 students. Must still return correct results without
    dividing by zero in college_avg_match_score computation.
    Referenced in sign-off report § Edge Cases.
    """

    def test_single_student_college_recommend(self):
        response = client.get("/recommend/college_D/student_D_0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) > 0
        # Every recommendation must have a reason string
        for rec in data["recommendations"]:
            assert "reason" in rec
            assert len(rec["reason"]) > 10

    def test_single_student_college_dashboard(self):
        response = client.get("/portal/college_D/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["enrolled_students"] >= 1
        # avg_match_score must be a valid float, not NaN
        assert not np.isnan(data["metrics"]["avg_match_score"])


# ── TEST 5: Unknown student at inference ─────────────────────────────────────

class TestUnknownStudentInference:
    """
    A student_id not in the training data or matching data must still receive
    a response (empty list with reason), not an error.
    Referenced in sign-off report § Edge Cases.
    """

    def test_unknown_student_returns_empty_not_error(self):
        response = client.get("/recommend/college_A/student_TOTALLY_UNKNOWN")
        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"] == []
        assert "No candidate" in data["reason"]


# ── TEST 6: College avg feature does not leak cross-college data ─────────────

class TestCollegeAvgFeatureIsolation:
    """
    Confirms that college_avg_match_score is a per-college aggregate computed
    at training time and does NOT require querying another college's records.
    """

    def test_college_avg_is_per_college(self):
        from src.ranking import add_derived_features

        df = pd.read_csv("data/matching_v1_output.csv")
        df_feat = add_derived_features(df)

        # For college_A rows, college_avg_match_score should equal college_A's own mean
        college_a = df_feat[df_feat["college_id"] == "college_A"]
        expected_avg = df[df["college_id"] == "college_A"]["match_score"].mean()
        actual_avg = college_a["college_avg_match_score"].iloc[0]
        assert abs(actual_avg - expected_avg) < 1e-6, (
            f"college_avg_match_score for college_A ({actual_avg}) "
            f"does not match college_A's own mean ({expected_avg})"
        )


# ── TEST 7: Recommendations have reason strings ─────────────────────────────

class TestRecommendationReasons:
    """
    Every recommendation must include a plain-English reason string.
    """

    def test_every_rec_has_reason(self):
        response = client.get("/recommend/college_A/student_A_0")
        assert response.status_code == 200
        for rec in response.json()["recommendations"]:
            assert "reason" in rec
            assert "Ranked #" in rec["reason"]
            assert "match_score" in rec["reason"]
