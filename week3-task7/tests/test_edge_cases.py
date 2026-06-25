import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from matcher import JobMatcher

@pytest.fixture
def matcher():
    # Will fallback to 0.0 model prob if model is missing, which is fine for tests
    # But ideally it will load the model we just trained
    return JobMatcher(model_path='models/logistic_regression.joblib')

def test_case_1_empty_candidate_profile(matcher):
    # Case 1: Empty Candidate Profile
    candidate = {}
    job = {'Job ID': 'J1', 'Required Skills': 'Python'}
    res = matcher.match(candidate, job)
    assert res['baseline_score'] == 0.0
    assert "Missing Python" in res['explanation']['why']
    assert res['candidate_id'] == 'Unknown'

def test_case_2_no_skills_found(matcher):
    # Case 2: No Skills Found
    candidate = {'Candidate ID': 'C1', 'Skills': ''}
    job = {'Job ID': 'J1', 'Required Skills': 'Python, SQL'}
    res = matcher.match(candidate, job)
    assert res['baseline_score'] == 0.0
    assert len(res['explanation']['missing_skills']) == 2

def test_case_3_duplicate_skills(matcher):
    # Case 3: Duplicate Skills
    candidate = {'Candidate ID': 'C1', 'Skills': 'Python, Python, SQL, sql'}
    job = {'Job ID': 'J1', 'Required Skills': 'Python, SQL'}
    res = matcher.match(candidate, job)
    # Baseline should still be 100% because set removes duplicates
    assert res['baseline_score'] == 100.0
    assert len(res['explanation']['matched_skills']) == 2

def test_case_4_job_missing_requirements(matcher):
    # Case 4: Job With Missing Requirements
    candidate = {'Candidate ID': 'C1', 'Skills': 'Python'}
    job = {'Job ID': 'J1'} # No required skills
    res = matcher.match(candidate, job)
    # If job has no requirements, baseline overlap should gracefully be 100%
    assert res['baseline_score'] == 100.0

def test_case_5_unknown_skills(matcher):
    # Case 5: Unknown Skills
    candidate = {'Candidate ID': 'C1', 'Skills': 'AlienTech, FutureScript'}
    job = {'Job ID': 'J1', 'Required Skills': 'Python, SQL'}
    res = matcher.match(candidate, job)
    assert res['baseline_score'] == 0.0

def test_case_6_experience_missing(matcher):
    # Case 6: Experience Missing
    candidate = {'Candidate ID': 'C1', 'Skills': 'Python'}
    # No Experience Years field
    job = {'Job ID': 'J1', 'Experience Requirement': 5}
    res = matcher.match(candidate, job)
    assert "[Missing] Missing experience (has 0.0y, needs 5.0y)" in res['explanation']['why']

def test_case_7_corrupted_input(matcher):
    # Case 7: Corrupted Input
    # Experience is a string instead of number, skills is a number
    candidate = {'Candidate ID': 'C1', 'Skills': 12345, 'Experience Years': 'five'}
    job = {'Job ID': 'J1', 'Required Skills': 'Python'}
    
    res = matcher.match(candidate, job)
    # The float conversions for 'five' will fallback to 0.0 without crashing
    assert res['baseline_score'] == 0.0

def test_case_8_candidate_matches_no_jobs(matcher):
    # Case 8: Candidate Matches No Jobs
    candidate = {'Candidate ID': 'C1', 'Skills': 'Ruby'}
    jobs = [
        {'Job ID': 'J1', 'Required Skills': 'Python'},
        {'Job ID': 'J2', 'Required Skills': 'Java'}
    ]
    results = matcher.rank_jobs(candidate, jobs)
    # Ensure it returns the list gracefully, even with low scores
    assert len(results) == 2
    assert results[0]['baseline_score'] == 0.0
    assert results[1]['baseline_score'] == 0.0
