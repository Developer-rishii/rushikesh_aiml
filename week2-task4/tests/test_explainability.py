import pytest
from src.explainability.explainer import ExplainerEngine

def test_explainability_engine():
    explainer = ExplainerEngine()
    
    student = {
        "student_id": 101,
        "skills": "Python,SQL,Machine Learning",
        "experience_years": 1,
        "verified_score": 85
    }
    
    job = {
        "job_id": 501,
        "required_skills": "Python,SQL,Communication",
        "minimum_score": 70
    }
    
    explanation = explainer.generate_explanation(student, job, match_score=87)
    
    assert explanation["match_score"] == 87
    assert set(explanation["matched_skills"]) == {"Python", "SQL"}
    assert set(explanation["missing_skills"]) == {"Communication"}
    
    reason = explanation["reason"]
    assert "satisfy 2 out of 3 required skills" in reason
    assert "exceeds the minimum qualification score" in reason
    assert "1 years of experience" in reason
    assert "is a strong match" in reason

def test_explainability_engine_not_match():
    explainer = ExplainerEngine()
    
    student = {
        "student_id": 102,
        "skills": "Java",
        "experience_years": 0,
        "verified_score": 50
    }
    
    job = {
        "job_id": 502,
        "required_skills": "Python,SQL",
        "minimum_score": 80
    }
    
    explanation = explainer.generate_explanation(student, job, match_score=20)
    
    assert explanation["match_score"] == 20
    assert len(explanation["matched_skills"]) == 0
    assert set(explanation["missing_skills"]) == {"Python", "SQL"}
    
    reason = explanation["reason"]
    assert "satisfy 0 out of 2 required skills" in reason
    assert "falls short of the minimum qualification score" in reason
    assert "is not a strong match" in reason
