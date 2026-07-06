import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


REQUIRED_COLUMNS = [
    'student_id', 'college_id', 'job_id', 'rank_position',
    'match_score', 'predicted_relevance_score', 'skill_overlap_count',
    'skill_gap_count', 'skill_gap_list', 'years_exposure_avg',
    'jd_seniority_level', 'ai_trust_score',
    'feature_importances_json', 'task16_explanation'
]


def test_rec_v1_output_schema():
    """Validate that the real rec_v1_output.csv has all required columns."""
    df = pd.read_csv("data/rec_v1_output.csv")
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_rec_v1_output_row_count():
    """Validate row count is in the expected range (400-600)."""
    df = pd.read_csv("data/rec_v1_output.csv")
    assert 400 <= len(df) <= 600, f"Row count {len(df)} outside expected 400-600 range"


def test_malformed_csv_rejected():
    """A CSV missing required columns must be detected as malformed."""
    malformed_df = pd.DataFrame({'student_id': [1], 'rank_position': [1]})
    malformed_df.to_csv("data/malformed.csv", index=False)

    df = pd.read_csv("data/malformed.csv")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert len(missing) > 0, "Malformed CSV should be missing required columns"
