import pytest
from src.match_vector import generate_match_vector
from src.threshold_validator import validate_thresholds
from src.scoring import calculate_match_score
from src.explainability import generate_explanation

@pytest.fixture
def sample_student():
    return {
        "python": 85,
        "machine_learning": 65,
        "sql": 90
    }

@pytest.fixture
def sample_job():
    return {
        "python_threshold": 80,
        "machine_learning_threshold": 70,
        "sql_threshold": 60
    }

def test_generate_match_vector(sample_student, sample_job):
    vector = generate_match_vector(sample_student, sample_job)
    # python: 85 >= 80 -> 1
    # ml: 65 < 70 -> 0
    # sql: 90 >= 60 -> 1
    assert vector == [1, 0, 1]

def test_validate_thresholds(sample_student, sample_job):
    validation = validate_thresholds(sample_student, sample_job)
    assert validation["eligible"] is False
    assert "Machine Learning" in validation["missing_skills"]
    assert "Python" not in validation["missing_skills"]

def test_calculate_match_score():
    vector = [1, 0, 1]
    score = calculate_match_score(vector)
    # 2/3 = 66.67
    assert score == 66.67
    
    empty_score = calculate_match_score([])
    assert empty_score == 0.0

def test_generate_explanation(sample_student, sample_job):
    vector = [1, 0, 1]
    score = 66.67
    eligible = False
    
    explanation = generate_explanation(sample_student, sample_job, score, eligible, vector)
    
    assert "Candidate matched 2 of 3 required skills." in explanation
    assert "Python:" in explanation
    assert "85 >= 80 [PASS]" in explanation
    assert "Machine Learning:" in explanation
    assert "65 < 70 [FAIL]" in explanation
    assert "SQL:" in explanation
    assert "90 >= 60 [PASS]" in explanation
    assert "Final Match Score: 66.67%" in explanation
    assert "Eligible: No" in explanation
