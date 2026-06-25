import os
import joblib
import numpy as np

from features import extract_features
from baseline import get_baseline_score

class JobMatcher:
    def __init__(self, model_path='../models/logistic_regression.joblib'):
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
    def match(self, candidate, job):
        """
        Calculates the baseline score and the model probability, 
        combines them into a final score, and generates an explanation.
        """
        # Baseline
        baseline_score = get_baseline_score(candidate, job)
        
        # Model Probability
        if self.model:
            feats = extract_features(candidate, job)
            X = np.array([[
                feats['skill_match_pct'],
                feats['req_skill_coverage'],
                feats['pref_skill_coverage'],
                feats['exp_match'],
                feats['edu_match'],
                feats['cert_match'],
                feats['project_relevance']
            ]])
            model_prob = self.model.predict_proba(X)[0][1]
        else:
            # Fallback if no model
            model_prob = 0.0
            
        # Final Score
        final_score = 0.7 * model_prob + 0.3 * baseline_score
        
        # Explainability
        explanation = self._generate_explanation(candidate, job, baseline_score, model_prob)
        
        return {
            'candidate_id': candidate.get('Candidate ID', 'Unknown'),
            'job_id': job.get('Job ID', 'Unknown'),
            'baseline_score': round(baseline_score * 100, 2),
            'model_probability': round(model_prob * 100, 2),
            'final_score': round(final_score * 100, 2),
            'explanation': explanation
        }
        
    def _generate_explanation(self, candidate, job, baseline, prob):
        c_skills = set([s.strip().lower() for s in str(candidate.get('Skills', '')).split(',') if s.strip()])
        j_req_skills = set([s.strip().lower() for s in str(job.get('Required Skills', '')).split(',') if s.strip()])
        j_pref_skills = set([s.strip().lower() for s in str(job.get('Preferred Skills', '')).split(',') if s.strip()])
        
        matched_req = c_skills.intersection(j_req_skills)
        missing_req = j_req_skills - c_skills
        
        try:
            c_exp = float(candidate.get('Experience Years', 0))
        except (ValueError, TypeError):
            c_exp = 0.0
            
        try:
            j_exp = float(job.get('Experience Requirement', 0))
        except (ValueError, TypeError):
            j_exp = 0.0
        
        reasons = []
        
        for skill in matched_req:
            reasons.append(f"[Match] {skill.title()} matched")
            
        if c_exp >= j_exp:
            reasons.append("[Match] Experience requirement met")
        else:
            reasons.append(f"[Missing] Missing experience (has {c_exp}y, needs {j_exp}y)")
            
        for skill in missing_req:
            reasons.append(f"[Missing] Missing {skill.title()}")
            
        return {
            'matched_skills': list(matched_req),
            'missing_skills': list(missing_req),
            'why': "\n".join(reasons)
        }

    def rank_jobs(self, candidate, jobs_list):
        """
        Ranks a list of jobs for a single candidate.
        """
        results = []
        for job in jobs_list:
            match_res = self.match(candidate, job)
            results.append(match_res)
            
        # Sort descending by final score
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking position
        for i, res in enumerate(results):
            res['ranking_position'] = i + 1
            
        return results
