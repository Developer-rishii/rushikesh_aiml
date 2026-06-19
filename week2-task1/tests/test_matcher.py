import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.matcher import calculate_match

def test_calculate_match_perfect():
    student = {
        'Skills': {'Python': 90, 'SQL': 90},
        'CGPA': 9.0,
        'Project Count': 2,
        'Internship Count': 1,
        'Communication Score': 85,
        'Aptitude Score': 85
    }
    job = {
        'Skill Requirements': {'Python': 80, 'SQL': 80},
        'Minimum CGPA': 8.0,
        'Experience Requirement': 1
    }
    
    score, details = calculate_match(student, job)
    
    # Expected: High score, meeting all reqs
    assert score >= 90
    assert details['skills']['total'] == 100.0 # Capped at 100
    assert details['experience']['student_units'] == 2.0 # 1 intern + 2*0.5 projects
    
def test_calculate_match_poor():
    student = {
        'Skills': {'Python': 60, 'SQL': 50},
        'CGPA': 6.0,
        'Project Count': 0,
        'Internship Count': 0,
        'Communication Score': 60,
        'Aptitude Score': 60
    }
    job = {
        'Skill Requirements': {'Python': 80, 'SQL': 80},
        'Minimum CGPA': 8.0,
        'Experience Requirement': 2
    }
    
    score, details = calculate_match(student, job)
    
    # Expected: Low score due to penalties
    assert score < 50
