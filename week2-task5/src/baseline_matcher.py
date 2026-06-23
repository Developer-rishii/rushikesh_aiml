import numpy as np
import ast

class BaselineMatcher:
    def __init__(self, threshold=70.0):
        self.threshold = threshold

    def _parse_skills(self, skills_str):
        if not skills_str or str(skills_str).lower() == 'nan':
            return []
        return [s.strip() for s in skills_str.split(',') if s.strip()]

    def match(self, job_series, student_series):
        # Edge Case 6: Missing candidate data
        if job_series is None or student_series is None or job_series.empty or student_series.empty:
            return {"error": "Missing job or candidate data"}

        job_req_skills = self._parse_skills(job_series.get('required_skills', ''))
        student_skills = self._parse_skills(student_series.get('verified_skills', ''))

        # Edge Case 2: Job has no skills listed
        if not job_req_skills:
            return {"error": "Validation error: Job has no required skills"}

        # Edge Case 1: Student has zero skills
        if not student_skills:
            match_vector = [0] * len(job_req_skills)
            match_score = 0.0
            reasons = [f"✗ {skill} missing" for skill in job_req_skills]
            average_verified_skill_score = 0.0
        else:
            match_vector = []
            reasons = []
            student_skills_lower = [s.lower() for s in student_skills]
            
            for skill in job_req_skills:
                if skill.lower() in student_skills_lower:
                    match_vector.append(1)
                    reasons.append(f"✓ {skill} found")
                else:
                    match_vector.append(0)
                    reasons.append(f"✗ {skill} missing")

            match_score = (sum(match_vector) / len(job_req_skills)) * 100

        # Parse skill scores to compute average
        try:
            skill_scores = ast.literal_eval(str(student_series.get('skill_scores', '{}')))
        except:
            skill_scores = {}
            
        # Calculate average of matched skills
        if student_skills:
            matched_skills = [s for s in job_req_skills if s.lower() in [st.lower() for st in student_skills]]
            scores_for_matched = [skill_scores.get(s, 0) for s in matched_skills]
            average_verified_skill_score = float(np.mean(scores_for_matched)) if scores_for_matched else 0.0

        min_skill_score = job_series.get('minimum_skill_score', 0)

        # Edge Case 4: Threshold boundary -> >= is passing
        status = "eligible" if (match_score >= self.threshold and average_verified_skill_score >= min_skill_score) else "rejected"
        
        explanation_header = "Candidate matched because:\n" if status == "eligible" else "Candidate rejected because:\n"
        explanation = explanation_header + "\n".join(reasons) + f"\nOverall Match Score = {match_score:.2f}%\nAverage Verified Skill Score = {average_verified_skill_score:.2f}"
        if average_verified_skill_score < min_skill_score and match_score >= self.threshold:
            explanation += f"\nRejected: Failed minimum skill score ({average_verified_skill_score:.2f} < {min_skill_score})"
        
        # Calculate experience gap for tiebreaker
        exp_gap = float(student_series.get('experience', 0)) - float(job_series.get('experience_required', 0))

        return {
            "job_id": job_series.get('job_id'),
            "student_id": student_series.get('student_id'),
            "match_vector": match_vector,
            "match_score": match_score,
            "average_verified_skill_score": average_verified_skill_score,
            "experience_gap": exp_gap,
            "threshold": self.threshold,
            "status": status,
            "explanation": explanation,
            "reasons_list": reasons
        }

if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Test cases
    matcher = BaselineMatcher(threshold=70.0)
    
    job_test = pd.Series({"job_id": "J001", "required_skills": "Python, SQL, Machine Learning"})
    student_test = pd.Series({"student_id": "S001", "verified_skills": "Python, SQL"})
    
    res = matcher.match(job_test, student_test)
    print("Match Result:")
    print(res["explanation"])
    print("Status:", res["status"])
