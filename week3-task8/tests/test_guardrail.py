import pytest
from src.guardrail import evaluate_guardrail
from src.threshold_calibration import calibrate
import pandas as pd

def test_zero_skill_overlap():
    match_data = {
        'prediction_score': 60.0,
        'candidate_skills': ['Python'],
        'job_skills': ['Java'],
        'candidate_exp': 2,
        'job_exp': 3
    }
    result = evaluate_guardrail(match_data, 50.0)
    assert result['fit_status'] == 'LOW_FIT_WARNING'
    assert any('Zero skill overlap' in r for r in result['reason'])

def test_good_match():
    match_data = {
        'prediction_score': 85.0,
        'candidate_skills': ['Python', 'SQL'],
        'job_skills': ['Python', 'SQL'],
        'candidate_exp': 3,
        'job_exp': 2
    }
    result = evaluate_guardrail(match_data, 50.0)
    assert result['fit_status'] == 'OK'

def test_calibration_small_dataset():
    df = pd.DataFrame({'is_success': [1]*10, 'prediction_score': [60]*10})
    with pytest.raises(ValueError, match="too small"):
        calibrate(df, log_path='dummy.csv')

def test_calibration_one_class():
    df = pd.DataFrame({'is_success': [1]*100, 'prediction_score': [60]*100})
    with pytest.raises(ValueError, match="both positive and negative samples"):
        calibrate(df, log_path='dummy.csv')
