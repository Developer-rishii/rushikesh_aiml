import pandas as pd
from baseline_matcher import BaselineMatcher
from ranker import CandidateRanker
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def test_edge_cases():
    matcher = BaselineMatcher(threshold=70.0)
    ranker = CandidateRanker(matcher)
    
    print("--- Edge Case 1: Student has zero skills ---")
    job_1 = pd.Series({"job_id": "J001", "required_skills": "Python, SQL"})
    student_1 = pd.Series({"student_id": "S001", "verified_skills": ""})
    res_1 = matcher.match(job_1, student_1)
    print(f"Match Score: {res_1['match_score']}")
    print(f"Status: {res_1['status']}")
    print()
    
    print("--- Edge Case 2: Job has no skills listed ---")
    job_2 = pd.Series({"job_id": "J002", "required_skills": ""})
    student_2 = pd.Series({"student_id": "S002", "verified_skills": "Python"})
    res_2 = matcher.match(job_2, student_2)
    print(res_2)
    print()
    
    print("--- Edge Case 3: Duplicate applications ---")
    job_3 = pd.Series({"job_id": "J003", "required_skills": "Python"})
    students_df_3 = pd.DataFrame([
        {"student_id": "S003", "verified_skills": "Python"},
        {"student_id": "S003", "verified_skills": "Python"} # Duplicate
    ])
    res_3 = ranker.rank_candidates(job_3, students_df_3)
    print(f"Number of candidates processed and shortlisted: {len(res_3)} (Expected 1)")
    print()
    
    print("--- Edge Case 4: Threshold boundary ---")
    job_4 = pd.Series({"job_id": "J004", "required_skills": "Python, SQL, AWS, Docker, Kubernetes"})
    # Match 3 out of 5 -> 60%
    # Match 4 out of 5 -> 80%
    # Let's test a matcher with threshold 80% to hit boundary exactly
    matcher_boundary = BaselineMatcher(threshold=80.0)
    student_4 = pd.Series({"student_id": "S004", "verified_skills": "Python, SQL, AWS, Docker"})
    res_4 = matcher_boundary.match(job_4, student_4)
    print(f"Threshold: {res_4['threshold']}")
    print(f"Score: {res_4['match_score']}")
    print(f"Status: {res_4['status']} (Expected eligible)")
    print()
    
    print("--- Edge Case 5: No candidates meet threshold ---")
    job_5 = pd.Series({"job_id": "J005", "required_skills": "Python, SQL"})
    students_df_5 = pd.DataFrame([
        {"student_id": "S005", "verified_skills": "Java"}
    ])
    res_5 = ranker.rank_candidates(job_5, students_df_5)
    print(f"Shortlist: {res_5} (Expected empty list [])")
    print()
    
    print("--- Edge Case 6: Missing candidate data ---")
    res_6 = matcher.match(job_1, None)
    print(res_6)
    print()

    print("--- Edge Case 7: Fails minimum skill score ---")
    job_7 = pd.Series({"job_id": "J007", "required_skills": "Python", "minimum_skill_score": 80.0})
    student_7 = pd.Series({"student_id": "S007", "verified_skills": "Python", "skill_scores": "{'Python': 70}"})
    res_7 = matcher.match(job_7, student_7)
    print(res_7["explanation"])
    print("Status:", res_7["status"], "(Expected rejected)")
    print("------------------------------------------")

if __name__ == "__main__":
    test_edge_cases()
