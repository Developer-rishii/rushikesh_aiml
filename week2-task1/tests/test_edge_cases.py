import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.matcher import calculate_match
from src.ranking import rank_candidates

def test_empty_skills_vs_requirements():
    student = {
        'Skills': {},
        'CGPA': 8.0,
        'Project Count': 2,
        'Internship Count': 1,
        'Communication Score': 85,
        'Aptitude Score': 85
    }
    job = {
        'Skill Requirements': {'Python': 80},
        'Minimum CGPA': 8.0,
        'Experience Requirement': 1
    }
    score, details = calculate_match(student, job)
    # Penalized because student doesn't have required skill
    assert details['skills']['total'] < 50
    assert score < 90

def test_empty_requirements():
    student = {
        'Skills': {'Python': 90},
        'CGPA': 8.0,
        'Project Count': 2,
        'Internship Count': 1,
        'Communication Score': 85,
        'Aptitude Score': 85
    }
    job = {
        'Skill Requirements': {},
        'Minimum CGPA': 8.0,
        'Experience Requirement': 1
    }
    score, details = calculate_match(student, job)
    # Final score shouldn't be penalized if other reqs met
    assert score >= 90

def test_malformed_dict():
    with pytest.raises(ValueError):
        calculate_match({}, {'Skill Requirements': {}})

def test_duplicate_ids():
    job = {
        'Skill Requirements': {'Python': 80},
        'Minimum CGPA': 8.0,
        'Experience Requirement': 1
    }
    # Two students with same ID
    students_df = pd.DataFrame([
        {'Student ID': 1, 'Name': 'Alice', 'Skills': {'Python': 90}, 'CGPA': 8.0, 'Project Count': 2, 'Internship Count': 1, 'Communication Score': 85, 'Aptitude Score': 85},
        {'Student ID': 1, 'Name': 'Alice Duplicate', 'Skills': {'Python': 90}, 'CGPA': 8.0, 'Project Count': 2, 'Internship Count': 1, 'Communication Score': 85, 'Aptitude Score': 85}
    ])
    ranked = rank_candidates(job, students_df)
    assert len(ranked) == 1
    assert ranked[0]['student_name'] == 'Alice'
