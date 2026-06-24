import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from baseline_matcher import calculate_baseline_score
from feature_engineering import extract_features_for_pair

def test_baseline_score_zero_overlap():
    score = calculate_baseline_score(['Python'], ['Java', 'SQL'])
    assert score == 0.0

def test_baseline_score_full_overlap():
    score = calculate_baseline_score(['Python', 'Java', 'SQL'], ['Java', 'SQL'])
    assert score == 100.0

def test_baseline_score_no_req_skills():
    score = calculate_baseline_score(['Python'], [])
    assert score == 100.0

def test_baseline_score_empty_candidate():
    score = calculate_baseline_score([], ['Java'])
    assert score == 0.0

def test_feature_extraction_missing_experience():
    c_dict = {'candidate_id': 1, 'skills': 'Python', 'years_experience': None, 'education': 'None'}
    j_dict = {'job_id': 1, 'required_skills': 'Python', 'minimum_experience': 2, 'preferred_education': 'None'}
    features = extract_features_for_pair(c_dict, j_dict)
    assert features['experience_gap'] == -2.0
    assert features['required_skill_coverage'] == 100.0
