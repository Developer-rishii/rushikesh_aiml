import pandas as pd

class BaselineMatcher:
    def __init__(self, candidates_path: str, jobs_path: str):
        self.candidates_df = pd.read_csv(candidates_path)
        self.jobs_df = pd.read_csv(jobs_path)
        self.candidates = self.candidates_df.set_index('candidate_id').to_dict('index')
        self.jobs = self.jobs_df.set_index('job_id').to_dict('index')

    def get_candidate(self, candidate_id: int):
        return self.candidates.get(candidate_id)

    def get_job(self, job_id: int):
        return self.jobs.get(job_id)

    def match(self, candidate_id: int, job_id: int) -> dict:
        c = self.get_candidate(candidate_id)
        j = self.get_job(job_id)
        if not c or not j:
            return None
        
        c_skills = set(c['skills'].split(',')) if pd.notna(c['skills']) else set()
        j_skills = set(j['required_skills'].split(',')) if pd.notna(j['required_skills']) else set()
        
        if len(j_skills) == 0:
            overlap = 1.0
        else:
            overlap = len(c_skills.intersection(j_skills)) / len(j_skills)
        
        return {
            "baseline_score": overlap * 100,
            "prediction_score": overlap * 100, # Simplified fallback
            "candidate_skills": list(c_skills),
            "job_skills": list(j_skills),
            "candidate_exp": c['experience_years'],
            "job_exp": j['min_experience']
        }
