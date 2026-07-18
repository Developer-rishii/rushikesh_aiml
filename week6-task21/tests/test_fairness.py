"""
test_fairness.py — pytest edge-case and integration tests for Task 21.
Run: pytest tests/ -v
All tests must pass before submission.
"""

import sys, json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fairness_metrics import compute_fairness_report, validate_input
from bias_classifier  import build_features, predict_one, MODEL_PATH


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def recs():
    return pd.read_csv(ROOT / "data" / "recommendations.csv")

@pytest.fixture(scope="module")
def labeled():
    return pd.read_csv(ROOT / "data" / "fairness_labels.csv")


# ── Bucket 4: dependency / failure / edge-case tests ──────────────────────────

def test_validate_input_missing_cols():
    """DEPENDENCY: malformed input raises ValueError, not a silent proceed."""
    bad = pd.DataFrame({"student_id": ["S001"], "college_tier": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input(bad)


def test_validate_input_empty():
    """DEPENDENCY: empty dataframe raises ValueError."""
    empty = pd.DataFrame(columns=["student_id", "college_tier", "region",
                                   "production_recommended", "recommended",
                                   "match_score_biased"])
    with pytest.raises(ValueError, match="empty"):
        validate_input(empty)


def test_validate_input_too_few_students():
    """DEPENDENCY: fewer than 10 students raises ValueError."""
    tiny = pd.DataFrame({
        "student_id":           [f"S{i}" for i in range(3)],
        "college_tier":         [1, 2, 3],
        "region":               ["urban"] * 3,
        "production_recommended": [1, 0, 1],
        "recommended":          [1, 1, 0],
        "match_score_biased":   [0.8, 0.6, 0.7],
    })
    with pytest.raises(ValueError, match="10 students"):
        validate_input(tiny)


def test_single_group_no_crash(recs):
    """EDGE CASE: single-group slice must not crash — DI trivially = 1.0."""
    single = recs[recs["college_tier"] == 1].copy()
    result = compute_fairness_report(single, ["college_tier"])
    assert "college_tier" in result
    assert result["college_tier"]["disparate_impact"] == 1.0


def test_predict_unknown_student(recs):
    """EDGE CASE: unknown student_id returns an error dict, not an exception."""
    result = predict_one("S9999_DOES_NOT_EXIST", "J000", recs)
    assert "error" in result
    assert "S9999_DOES_NOT_EXIST" in result["error"]


def test_perfect_bias_detected():
    """EDGE CASE: perfect disparity (DI=0.0) is detected and flagged correctly."""
    n = 100
    df = pd.DataFrame({
        "student_id":           [f"S{i}" for i in range(n)],
        "college_tier":         [1] * 50 + [3] * 50,
        "region":               ["urban"] * n,
        "production_recommended": [1] * 50 + [0] * 50,  # tier 3 gets zero recs
        "recommended":          [1] * n,                 # skill says all good
        "match_score_biased":   [0.9] * 50 + [0.4] * 50,
    })
    result = compute_fairness_report(df, ["college_tier"])
    assert result["college_tier"]["fails_4_5ths_rule"] is True
    assert result["college_tier"]["disparate_impact"] == 0.0


def test_disparate_impact_formula(recs):
    """CORRECTNESS: disparate_impact = min_rate / max_rate across groups."""
    result = compute_fairness_report(recs, ["college_tier"])
    stats  = result["college_tier"]["group_stats"]
    rates  = [row["rec_rate"] for row in stats]
    expected_di = min(rates) / max(rates)
    assert abs(result["college_tier"]["disparate_impact"] - expected_di) < 1e-4


def test_model_artifact_exists():
    """DEPENDENCY: model artifact must exist after training."""
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"


def test_predict_returns_reason(recs):
    """EXPLAINABILITY: every prediction must include a non-empty reason string."""
    first_student = recs["student_id"].iloc[0]
    first_job     = recs[recs["student_id"] == first_student]["job_id"].iloc[0]
    result = predict_one(first_student, first_job, recs)
    assert "reason" in result
    assert len(result["reason"]) > 20
    assert "verdict" in result
    assert result["verdict"] in ("⚠️ BIAS RISK", "✅ SKILL-DRIVEN")


def test_bias_score_range(recs):
    """CORRECTNESS: bias_risk_score must be in [0, 1]."""
    first_student = recs["student_id"].iloc[0]
    first_job     = recs[recs["student_id"] == first_student]["job_id"].iloc[0]
    result = predict_one(first_student, first_job, recs)
    assert 0.0 <= result["bias_risk_score"] <= 1.0


def test_tier3_rural_higher_bias_risk(recs):
    """
    PROOF OF DETECTION: Tier3 rural students should have higher avg bias_risk
    than Tier1 urban students — because bias was injected in data generation.
    """
    tier1_urban = recs[(recs["college_tier"] == 1) & (recs["region"] == "urban")]
    tier3_rural = recs[(recs["college_tier"] == 3) & (recs["region"] == "rural")]

    import joblib
    from bias_classifier import build_features, MODEL_PATH
    artifact = joblib.load(MODEL_PATH)
    clf, threshold, feature_cols = (artifact["model"], artifact["threshold"],
                                     artifact["feature_cols"])

    def score_group(df_group):
        from bias_classifier import build_features
        df_group = df_group.copy()
        df_group["is_biased_outcome"] = 0  # dummy — not used in features
        df_group["skill_only_recommended"] = df_group["recommended"]
        df_group["production_recommended_col"] = df_group["production_recommended"]
        features = build_features(df_group)[feature_cols]
        return clf.predict_proba(features)[:, 1].mean()

    avg_t1 = score_group(tier1_urban.head(50))
    avg_t3 = score_group(tier3_rural.head(50))
    # Tier3 rural should have meaningfully higher avg bias risk
    assert avg_t3 > avg_t1, (
        f"Expected Tier3 rural avg bias score ({avg_t3:.3f}) > "
        f"Tier1 urban ({avg_t1:.3f}) — bias injection not being detected"
    )


def test_experiment_log_written():
    """REPRODUCIBILITY: experiment log must be written after training."""
    log_path = ROOT / "reports" / "experiment_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) >= 1
    assert "precision" in entries[-1]
    assert "threshold" in entries[-1]


def test_fairness_report_saved():
    """EVIDENCE: fairness_report.json must be saved after evaluate.py runs."""
    report_path = ROOT / "reports" / "fairness_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "baseline" in report
    assert "ml_classifier" in report
    assert "college_tier" in report["baseline"]
    assert "region" in report["baseline"]
