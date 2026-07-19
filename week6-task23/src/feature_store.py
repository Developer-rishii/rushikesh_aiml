"""
src/feature_store.py

A lightweight local Feature Store for PlaceMux.
Stores student and job features (profiles) for fast online retrieval during inference,
and provides historical feature joining for training.
"""
import os
import pandas as pd
from typing import Dict, Any, Tuple
from src.features import build_feature_matrix, _safe_json_load, build_features_row

class FeatureStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._students_df = None
        self._jobs_df = None
        self._s_idx = None
        self._j_idx = None

    def _load_data(self):
        if self._students_df is None:
            self._students_df = pd.read_csv(os.path.join(self.data_dir, "students.csv"))
            self._s_idx = self._students_df.set_index("student_id")
        if self._jobs_df is None:
            self._jobs_df = pd.read_csv(os.path.join(self.data_dir, "jobs.csv"))
            self._j_idx = self._jobs_df.set_index("job_id")

    def get_online_features(self, student_id: str, job_id: str) -> Dict[str, Any]:
        """Fetch pre-computed entity features and combine them for online inference."""
        self._load_data()
        
        if student_id not in self._s_idx.index:
            raise ValueError(f"Student {student_id} not found in feature store.")
        if job_id not in self._j_idx.index:
            raise ValueError(f"Job {job_id} not found in feature store.")
            
        student_row = self._s_idx.loc[student_id]
        job_row = self._j_idx.loc[job_id]
        
        s_skills = _safe_json_load(student_row["skills_json"], "skills_json", student_id)
        j_skills = _safe_json_load(job_row["required_skills_json"], "required_skills_json", job_id)
        years_gap = int(student_row["years_experience"]) - int(job_row["years_required"])
        
        feat = build_features_row(s_skills, j_skills, years_gap)
        return feat

    def get_historical_features(self, interactions: pd.DataFrame, drop_bad_rows: bool = True) -> pd.DataFrame:
        """Point-in-time join for training data."""
        self._load_data()
        return build_feature_matrix(interactions, self._students_df, self._jobs_df, drop_bad_rows=drop_bad_rows)
