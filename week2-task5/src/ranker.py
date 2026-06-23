import pandas as pd
from baseline_matcher import BaselineMatcher

class CandidateRanker:
    def __init__(self, matcher):
        self.matcher = matcher

    def rank_candidates(self, job_series, students_df):
        # Edge Case 6: Missing candidate data
        if job_series is None or students_df is None or students_df.empty:
            return {"error": "Missing job or candidate data"}

        # Edge Case 3: Duplicate applications
        # Drop duplicates based on student_id
        unique_students = students_df.drop_duplicates(subset=['student_id'])
        
        results = []
        for _, student in unique_students.iterrows():
            match_res = self.matcher.match(job_series, student)
            if "error" in match_res:
                continue # Skip invalid matches
                
            results.append(match_res)
            
        # Filter eligible
        shortlist = [res for res in results if res["status"] == "eligible"]
        
        # Edge Case 5: No candidates meet threshold
        if not shortlist:
            return [] # Return empty shortlist
            
        # Rank by match score descending. 
        # Tiebreak Rule:
        # 1. average_verified_skill_score (descending): candidates with higher mastery of matched skills rank higher
        # 2. experience_gap (descending): candidates with more surplus experience over the requirement rank higher
        shortlist_sorted = sorted(shortlist, key=lambda x: (x["match_score"], x.get("average_verified_skill_score", 0.0), x.get("experience_gap", 0.0)), reverse=True)
        return shortlist_sorted
