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
        
        c_skills = set(str(c.get('skills', '')).split(',')) if pd.notna(c.get('skills', '')) and c.get('skills', '') else set()
        j_skills = set(str(j.get('required_skills', '')).split(',')) if pd.notna(j.get('required_skills', '')) and j.get('required_skills', '') else set()
        
        # Clean empty strings
        c_skills.discard('')
        j_skills.discard('')
        
        if len(j_skills) == 0:
            skill_overlap = 1.0
        else:
            skill_overlap = len(c_skills.intersection(j_skills)) / len(j_skills)
        
        exp_gap = float(c.get('experience_years', 0)) - float(j.get('min_experience', 0))
        
        edu_match = 1 if c.get('education', '') == j.get('required_education', '') else 0
        
        c_certs = set(str(c.get('certifications', '')).split(',')) if pd.notna(c.get('certifications', '')) and c.get('certifications', '') else set()
        j_certs = set(str(j.get('required_certifications', '')).split(',')) if pd.notna(j.get('required_certifications', '')) and j.get('required_certifications', '') else set()
        c_certs.discard('')
        j_certs.discard('')
        cert_match_count = len(c_certs.intersection(j_certs))
        
        # Compute prediction_score using the real model
        import pickle
        import numpy as np
        model_path = 'd:/Placemux-aiml/week3-task8/models/baseline_model.pkl'
        try:
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)
            features = np.array([[skill_overlap, exp_gap, edu_match, cert_match_count]])
            prediction_score = round(clf.predict_proba(features)[0][1] * 100, 2)
        except Exception:
            prediction_score = round(skill_overlap * 100, 2)
        
        return {
            "baseline_score": round(skill_overlap * 100, 2),
            "prediction_score": prediction_score,
            "candidate_skills": list(c_skills),
            "job_skills": list(j_skills),
            "candidate_exp": float(c.get('experience_years', 0)),
            "job_exp": float(j.get('min_experience', 0)),
            "edu_match": edu_match,
            "cert_match_count": cert_match_count
        }
