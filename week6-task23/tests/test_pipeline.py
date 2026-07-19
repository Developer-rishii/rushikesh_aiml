"""
tests/test_pipeline.py

Dependency, failure & edge-case handling (15 marks in the rubric).
Run: pytest -v tests/test_pipeline.py
"""
import os
import sys
import json
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import build_features_row, build_feature_matrix, FeatureError
from src.model import MatchModel
from src.drift_detector import compute_drift_report, PSI_RETRAIN_THRESHOLD


# ---------- Feature engineering edge cases ----------

def test_empty_job_skills_no_crash():
    """Cold-start JD with zero requirements must not crash -- returns a
    valid, low-confidence feature row instead."""
    f = build_features_row({"python": 0.8}, {}, years_gap=1)
    assert f["overlap_count"] == 0
    assert f["missing_top_skill"] == 1
    assert f["overlap_ratio"] == 0.0


def test_empty_student_skills_no_crash():
    """Cold-start student (just signed up, nothing verified yet)."""
    f = build_features_row({}, {"python": 0.9}, years_gap=-2)
    assert f["overlap_count"] == 0
    assert f["missing_top_skill"] == 1


def test_full_overlap_perfect_match():
    f = build_features_row({"python": 1.0, "sql": 1.0}, {"python": 1.0, "sql": 1.0}, years_gap=0)
    assert f["overlap_ratio"] == 1.0
    assert f["missing_top_skill"] == 0
    assert f["weighted_skill_score"] == pytest.approx(1.0)


def test_malformed_skills_json_is_skipped_not_crashed():
    """A malformed row (bad JSON) must be logged & dropped, not blow up
    the whole batch -- 'errors handled' requirement."""
    students = pd.DataFrame([
        {"student_id": "S1", "years_experience": 3, "skills_json": '{"python": 0.8'},  # truncated/broken JSON
        {"student_id": "S2", "years_experience": 3, "skills_json": json.dumps({"python": 0.8})},
    ])
    jobs = pd.DataFrame([{"job_id": "J1", "years_required": 2,
                           "required_skills_json": json.dumps({"python": 0.9})}])
    interactions = pd.DataFrame([
        {"student_id": "S1", "job_id": "J1", "month": "2026-01", "good_match": 1},
        {"student_id": "S2", "job_id": "J1", "month": "2026-01", "good_match": 1},
    ])
    result = build_feature_matrix(interactions, students, jobs, drop_bad_rows=True)
    assert len(result) == 1  # only S2 survives
    assert result.iloc[0]["student_id"] == "S2"


def test_malformed_row_raises_when_not_dropping():
    students = pd.DataFrame([{"student_id": "S1", "years_experience": 3, "skills_json": "not json"}])
    jobs = pd.DataFrame([{"job_id": "J1", "years_required": 2,
                           "required_skills_json": json.dumps({"python": 0.9})}])
    interactions = pd.DataFrame([{"student_id": "S1", "job_id": "J1", "good_match": 1}])
    with pytest.raises(FeatureError):
        build_feature_matrix(interactions, students, jobs, drop_bad_rows=False)


def test_unknown_student_id_handled():
    students = pd.DataFrame([{"student_id": "S1", "years_experience": 3,
                               "skills_json": json.dumps({"python": 0.8})}])
    jobs = pd.DataFrame([{"job_id": "J1", "years_required": 2,
                           "required_skills_json": json.dumps({"python": 0.9})}])
    interactions = pd.DataFrame([{"student_id": "GHOST", "job_id": "J1", "good_match": 1}])
    result = build_feature_matrix(interactions, students, jobs, drop_bad_rows=True)
    assert len(result) == 0  # dropped, not crashed


# ---------- Model edge cases ----------

def test_model_predicts_on_single_row():
    students = pd.DataFrame([{"student_id": f"S{i}", "years_experience": 3,
                               "skills_json": json.dumps({"python": 0.8, "sql": 0.7})} for i in range(20)])
    jobs = pd.DataFrame([{"job_id": f"J{i}", "years_required": 2,
                           "required_skills_json": json.dumps({"python": 0.9})} for i in range(20)])
    interactions = pd.DataFrame([{"student_id": f"S{i}", "job_id": f"J{i}",
                                   "good_match": i % 2} for i in range(20)])
    feats = build_feature_matrix(interactions, students, jobs)
    model = MatchModel("test").fit(feats)
    one_row = feats.iloc[[0]]
    proba = model.predict_proba(one_row)
    assert 0.0 <= proba[0] <= 1.0


# ---------- Drift detector edge cases ----------

def test_drift_report_identical_distributions_is_stable():
    df = pd.DataFrame({"overlap_count": [1, 2, 3, 4, 5] * 20,
                        "overlap_ratio": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
                        "weighted_skill_score": [0.5] * 100,
                        "years_gap": [0] * 100, "missing_top_skill": [0] * 100,
                        "jd_breadth": [5] * 100, "student_breadth": [8] * 100})
    report = compute_drift_report(df, df.copy(), list(df.columns))
    assert report["status"] == "STABLE"
    assert report["trigger_retrain"] is False


def test_drift_report_empty_current_batch_no_crash():
    ref = pd.DataFrame({"overlap_count": [1, 2, 3]})
    cur = pd.DataFrame({"overlap_count": []})
    report = compute_drift_report(ref, cur, ["overlap_count"])
    assert report["trigger_retrain"] is False  # no data = no false alarm
