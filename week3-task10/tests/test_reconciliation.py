"""
PlaceMux Quality Sign-Off - Automated Tests
=============================================
Pytest suite covering all edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from src.reconciliation import (
    reconcile_payments,
    handle_payment_failure,
    validate_amounts,
)
from src.baseline import compute_baseline_score, parse_required_skills
from src.features import build_features
from src.labeling import compute_label

@pytest.fixture
def sample_student():
    return pd.Series({
        "student_id": "S001",
        "Python": 4, "JavaScript": 3, "React": 5, "Node.js": 2,
        "SQL": 4, "Docker": np.nan, "AWS": 1, "Machine Learning": 3,
        "years_of_exposure": 5.0,
    })

@pytest.fixture
def sample_job():
    return pd.Series({
        "job_id": "J001",
        "required_skills": "Python|React|SQL|Docker",
        "required_levels": "Python:3|React:4|SQL:3|Docker:2",
        "price_tier": "basic",
    })

@pytest.fixture
def zero_overlap_job():
    return pd.Series({
        "job_id": "J999",
        "required_skills": "Cobol|Fortran|VHDL",
        "required_levels": "Cobol:3|Fortran:2|VHDL:4",
        "price_tier": "free",
    })

@pytest.fixture
def sample_events():
    return pd.DataFrame([
        {"application_id": "A001", "student_id": "S001", "job_id": "J001",
         "payment_status": "success", "gateway_amount": 29.99, "recorded_amount": 29.99},
        {"application_id": "A002", "student_id": "S002", "job_id": "J001",
         "payment_status": "failed", "gateway_amount": 29.99, "recorded_amount": 29.99},
        {"application_id": "A003", "student_id": "S001", "job_id": "J002",
         "payment_status": "success", "gateway_amount": 99.99, "recorded_amount": 89.99},
        {"application_id": "A004", "student_id": "S001", "job_id": "J001",
         "payment_status": "pending", "gateway_amount": 29.99, "recorded_amount": 0.0},
        {"application_id": "A005", "student_id": "S001", "job_id": "J001",
         "payment_status": "success", "gateway_amount": 29.99, "recorded_amount": 29.99},
    ])

class TestHappyPath:
    def test_successful_payment_reconciles_clean(self):
        events = pd.DataFrame([{
            "application_id": "A100", "student_id": "S001", "job_id": "J001",
            "payment_status": "success", "gateway_amount": 29.99, "recorded_amount": 29.99,
        }])
        result = reconcile_payments(events)
        assert len(result["amount_mismatches"]) == 0
        assert len(result["failed_applications"]) == 0

    def test_handle_success_records_match(self):
        event = {"application_id": "A100", "student_id": "S001", "job_id": "J001",
                 "payment_status": "success", "gateway_amount": 29.99}
        result = handle_payment_failure(event)
        assert result["match_recorded"] is True
        assert result["application_retained"] is True
        assert result["status"] == "active"

class TestPaymentFailure:
    def test_student_retains_application_on_failure(self):
        event = {"application_id": "A200", "student_id": "S001", "job_id": "J001",
                 "payment_status": "failed", "gateway_amount": 29.99}
        result = handle_payment_failure(event)
        assert result["application_retained"] is True
        assert result["match_recorded"] is False

    def test_refund_initiated_on_charged_failure(self):
        event = {"application_id": "A201", "student_id": "S001", "job_id": "J001",
                 "payment_status": "failed", "gateway_amount": 99.99}
        result = handle_payment_failure(event)
        assert result["refund_initiated"] is True
        assert result["refund_amount"] == 99.99
        assert result["status"] == "refund_pending"

    def test_no_refund_on_free_failure(self):
        event = {"application_id": "A202", "student_id": "S001", "job_id": "J001",
                 "payment_status": "failed", "gateway_amount": 0.0}
        result = handle_payment_failure(event)
        assert result["refund_initiated"] is False

    def test_pending_payment_retains_application(self):
        event = {"application_id": "A203", "student_id": "S001", "job_id": "J001",
                 "payment_status": "pending", "gateway_amount": 29.99}
        result = handle_payment_failure(event)
        assert result["application_retained"] is True
        assert result["status"] == "payment_pending"

    def test_failed_apps_flagged_in_reconciliation(self, sample_events):
        result = reconcile_payments(sample_events)
        failed = result["failed_applications"]
        assert len(failed) >= 1

class TestAmountMismatch:
    def test_mismatch_detected(self, sample_events):
        result = reconcile_payments(sample_events)
        mismatches = result["amount_mismatches"]
        assert len(mismatches) >= 1
        mismatch_ids = [m["application_id"] for m in mismatches]
        assert "A003" in mismatch_ids

    def test_mismatch_severity(self, sample_events):
        result = reconcile_payments(sample_events)
        for m in result["amount_mismatches"]:
            if m["application_id"] == "A003":
                assert m["severity"] == "HIGH"

    def test_validate_amounts_catches_discrepancy(self, sample_events):
        bad = validate_amounts(sample_events)
        assert len(bad) >= 1
        assert "A003" in bad["application_id"].values

    def test_matching_amounts_pass(self):
        events = pd.DataFrame([{
            "application_id": "A300", "student_id": "S001", "job_id": "J001",
            "payment_status": "success", "gateway_amount": 29.99, "recorded_amount": 29.99,
        }])
        bad = validate_amounts(events)
        assert len(bad) == 0

class TestMissingSkillData:
    def test_baseline_handles_nan_skills(self):
        student = pd.Series({
            "Python": np.nan, "JavaScript": np.nan, "React": np.nan,
            "Node.js": np.nan, "SQL": 3, "Docker": np.nan, "AWS": np.nan,
            "Machine Learning": np.nan, "years_of_exposure": 1.0,
        })
        job = pd.Series({
            "required_skills": "Python|SQL",
            "required_levels": "Python:3|SQL:2",
            "price_tier": "free",
        })
        score = compute_baseline_score(student, job)
        assert score["overlap_count"] == 1
        assert score["total_required"] == 2
        assert score["overlap_ratio"] == 0.5

    def test_features_handle_nan_skills(self):
        student = pd.Series({
            "Python": np.nan, "JavaScript": np.nan, "React": np.nan,
            "Node.js": np.nan, "SQL": 3, "Docker": np.nan, "AWS": np.nan,
            "Machine Learning": np.nan, "years_of_exposure": 1.0,
        })
        job = pd.Series({
            "required_skills": "Python|SQL",
            "required_levels": "Python:3|SQL:2",
            "price_tier": "free",
        })
        event = pd.Series({"payment_status": "success"})
        feats = build_features(student, job, event)
        assert feats["skill_overlap_count"] == 1
        assert feats["num_missing_required"] == 1

    def test_label_with_all_nan_skills(self):
        student = pd.Series({
            "Python": np.nan, "JavaScript": np.nan, "React": np.nan,
            "Node.js": np.nan, "SQL": np.nan, "Docker": np.nan,
            "AWS": np.nan, "Machine Learning": np.nan, "years_of_exposure": 0.5,
        })
        job = pd.Series({
            "required_skills": "Python|SQL",
            "required_levels": "Python:3|SQL:2",
            "price_tier": "free",
        })
        label = compute_label(student, job)
        assert label == 0

class TestZeroOverlapJD:
    def test_baseline_zero_overlap(self, sample_student, zero_overlap_job):
        score = compute_baseline_score(sample_student, zero_overlap_job)
        assert score["overlap_count"] == 0
        assert score["is_match"] == 0

    def test_label_zero_overlap(self, sample_student, zero_overlap_job):
        label = compute_label(sample_student, zero_overlap_job)
        assert label == 0

    def test_features_zero_overlap(self, sample_student, zero_overlap_job):
        event = pd.Series({"payment_status": "success"})
        feats = build_features(sample_student, zero_overlap_job, event)
        assert feats["skill_overlap_count"] == 0
        assert feats["num_missing_required"] == 3

class TestDuplicatePayments:
    def test_duplicates_detected(self, sample_events):
        result = reconcile_payments(sample_events)
        dupes = result["duplicate_events"]
        assert len(dupes) >= 1

    def test_partial_payment_flagged(self, sample_events):
        result = reconcile_payments(sample_events)
        mismatches = result["amount_mismatches"]
        partial_ids = [m["application_id"] for m in mismatches]
        assert "A004" in partial_ids

    def test_charged_without_match_flagged(self, sample_events):
        result = reconcile_payments(sample_events)
        charged = result["charged_without_match"]
        assert len(charged) >= 1

class TestBaselineCorrectness:
    def test_full_overlap(self, sample_student, sample_job):
        score = compute_baseline_score(sample_student, sample_job)
        assert score["overlap_count"] == 3
        assert score["total_required"] == 4
        assert score["overlap_ratio"] == 0.75
        assert score["is_match"] == 1

    def test_parse_required_skills(self, sample_job):
        skills = parse_required_skills(sample_job)
        assert "Python" in skills
        assert "Docker" in skills
        assert len(skills) == 4

class TestFeatureEdgeCases:
    def test_payment_one_hot(self, sample_student, sample_job):
        event = pd.Series({"payment_status": "refunded"})
        feats = build_features(sample_student, sample_job, event)
        assert feats["pay_refunded"] == 1
        assert feats["pay_success"] == 0
        assert feats["pay_failed"] == 0

    def test_price_tier_one_hot(self, sample_student, sample_job):
        event = pd.Series({"payment_status": "success"})
        feats = build_features(sample_student, sample_job, event)
        assert feats["price_tier_basic"] == 1
        assert feats["price_tier_free"] == 0
        assert feats["price_tier_premium"] == 0

    def test_coverage_score_bounded(self, sample_student, sample_job):
        event = pd.Series({"payment_status": "success"})
        feats = build_features(sample_student, sample_job, event)
        assert 0.0 <= feats["weighted_coverage_score"] <= 1.0
