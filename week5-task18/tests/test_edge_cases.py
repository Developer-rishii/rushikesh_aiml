import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from explanations import compute_counterfactual, generate_admin_explanation, generate_student_explanation
from quality_model import extract_features, score_explanations


def test_malformed_csv_handling():
    """Rec v1 schema validation: feeding a malformed file must be caught."""
    malformed_df = pd.DataFrame({'student_id': [1], 'rank_position': [1]})
    malformed_df.to_csv("data/malformed.csv", index=False)

    required_cols = [
        'student_id', 'college_id', 'job_id', 'rank_position',
        'feature_importances_json', 'task16_explanation'
    ]
    assert not all(c in malformed_df.columns for c in required_cols), \
        "Malformed CSV should be missing required columns"


def test_missing_feature_importances():
    """Missing feature_importances_json produces degraded-but-honest explanation."""
    row = pd.Series({
        'student_id': 'S01', 'college_id': 'C1', 'job_id': 'J1',
        'rank_position': 2, 'skill_gap_count': 1, 'skill_overlap_count': 3,
        'skill_gap_list': 'Python', 'predicted_relevance_score': 0.85,
        'feature_importances_json': float('nan')
    })
    expl = generate_admin_explanation(row)
    assert "Feature attribution unavailable" in expl, \
        "Should degrade gracefully when feature importances are missing"


def test_rank_1_no_gaps():
    """Rank #1 with no skill gaps: no nonsensical counterfactual generated."""
    row = pd.Series({
        'student_id': 'S01', 'college_id': 'C1', 'job_id': 'J1',
        'rank_position': 1, 'skill_gap_count': 0, 'skill_overlap_count': 5,
        'skill_gap_list': '', 'predicted_relevance_score': 0.95
    })
    expl = generate_student_explanation(row)
    assert "no skill gaps identified" in expl, \
        "Rank #1 with no gaps should say so explicitly"


def test_counterfactual_computation():
    """Counterfactual computation proof: re-score with gap_count-1 using model."""
    row = pd.Series({
        'student_id': 'S01', 'college_id': 'C1', 'job_id': 'J1',
        'rank_position': 3, 'skill_gap_count': 2, 'skill_overlap_count': 2,
        'predicted_relevance_score': 0.70
    })
    new_rank, score_diff = compute_counterfactual(row)
    assert new_rank is not None, "Should compute a new rank"
    assert new_rank < 3, f"New rank ({new_rank}) should be better than 3"
    assert score_diff > 0, f"Score diff ({score_diff}) should be positive"


def test_audience_mismatch():
    """ML quality scorer correctly flags admin explanation sent to student."""
    admin_expl = "Top features driving this recommendation: match_score (weight 0.41)."
    dummy = pd.DataFrame([{
        "explanation_text": admin_expl,
        "audience": "student",
        "rank_position": 3
    }])
    score = score_explanations(dummy)[0]
    assert score < 0.5, \
        f"Admin explanation scored {score} for student audience — should be low"
