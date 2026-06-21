import pandas as pd
import pickle
from .feature_engineering import compute_features
from .explainability import generate_explanation

class Ranker:
    def __init__(self, model_path="models/model.pkl", students_path="data/students.csv", jobs_path="data/jobs.csv"):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.students = pd.read_csv(students_path).set_index('student_id')
        self.jobs = pd.read_csv(jobs_path).set_index('job_id')
        
    def rank_jobs_for_student(self, student_id, top_n=5):
        if student_id not in self.students.index:
            return {"error": "Student not found."}
            
        student_data = self.students.loc[[student_id]]
        
        # Create a cross product of this student and all jobs
        # Pandas 1.2+ supports cross merge
        df_cross = pd.merge(student_data.reset_index(), self.jobs.reset_index(), how='cross')
        
        # Rename location columns appropriately because both have 'location'
        df_cross = df_cross.rename(columns={'location_x': 'location_x', 'location_y': 'location_y'})
        if 'location' in df_cross.columns:
            # If cross didn't rename, do it manually
            pass 
        
        features_df = compute_features(None, None, df_cross)
        
        # Predict probability
        features_only = features_df[['skill_overlap', 'experience_match', 'education_match', 'location_match']]
        probs = self.model.predict_proba(features_only)[:, 1]
        
        df_cross['score'] = probs
        df_sorted = df_cross.sort_values(by='score', ascending=False).head(top_n)
        
        results = []
        for _, row in df_sorted.iterrows():
            explanation = generate_explanation(
                student_row=self.students.loc[student_id], 
                job_row=self.jobs.loc[row['job_id']], 
                score=row['score']
            )
            results.append({
                "job_id": row['job_id'],
                "job": row['job_title'],
                "score": explanation['match_score'],
                "explanation": explanation['reasons']
            })
            
        return results

    def rank_candidates_for_job(self, job_id, top_n=5):
        if job_id not in self.jobs.index:
            return {"error": "Job not found."}
            
        job_data = self.jobs.loc[[job_id]]
        
        # Create cross product
        df_cross = pd.merge(self.students.reset_index(), job_data.reset_index(), how='cross')
        
        features_df = compute_features(None, None, df_cross)
        
        features_only = features_df[['skill_overlap', 'experience_match', 'education_match', 'location_match']]
        probs = self.model.predict_proba(features_only)[:, 1]
        
        df_cross['score'] = probs
        df_sorted = df_cross.sort_values(by='score', ascending=False).head(top_n)
        
        results = []
        for _, row in df_sorted.iterrows():
            explanation = generate_explanation(
                student_row=self.students.loc[row['student_id']], 
                job_row=self.jobs.loc[job_id], 
                score=row['score']
            )
            results.append({
                "candidate_id": row['student_id'],
                "candidate": row['name'],
                "score": explanation['match_score'],
                "explanation": explanation['reasons']
            })
            
        return results
