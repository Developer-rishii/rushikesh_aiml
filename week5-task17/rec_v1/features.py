"""
features.py

Feature engineering pipeline.
Constructs ML-ready features from the raw generated datasets.
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.priors = {}
        self.student_map = {}
        self.job_map = {}
        self.student_skills_map = {}
        self.job_skills_map = {}
        
    def load_context(self, data_dir="data"):
        """Load context tables for fast feature lookup"""
        students_df = pd.read_csv(f"{data_dir}/students.csv")
        self.student_map = students_df.set_index("student_id").to_dict(orient="index")
        
        jobs_df = pd.read_csv(f"{data_dir}/jobs.csv")
        self.job_map = jobs_df.set_index("job_id").to_dict(orient="index")
        
        # Deduplicate student skills by taking max proficiency (handles duplicates edge case)
        student_skills = pd.read_csv(f"{data_dir}/student_skills.csv")
        student_skills = student_skills.groupby(["student_id", "skill_id"])["proficiency"].max().reset_index()
        
        self.student_skills_map = student_skills.groupby("student_id").apply(
            lambda x: dict(zip(x["skill_id"], x["proficiency"]))
        ).to_dict()
        
        job_skills = pd.read_csv(f"{data_dir}/job_skills.csv")
        self.job_skills_map = job_skills.groupby("job_id").apply(
            lambda x: dict(zip(x["skill_id"], x["min_proficiency"]))
        ).to_dict()

    def fit(self, train_outcomes_df):
        """Learn priors from training data only to avoid leakage."""
        # Calculate historical hire rate per college
        merged = train_outcomes_df.copy()
        merged["college_id"] = merged["student_id"].map(lambda x: self.student_map.get(x, {}).get("college_id"))
        
        # Prior 1: College historical hire rate
        self.priors["college_hire_rate"] = merged.groupby("college_id")["was_hired"].mean().to_dict()
        self.priors["global_hire_rate"] = merged["was_hired"].mean()

    def transform(self, pairs_df):
        """
        Takes a DataFrame of (student_id, job_id) pairs.
        Returns a DataFrame of features.
        """
        features = []
        for _, row in pairs_df.iterrows():
            sid = row["student_id"]
            jid = row["job_id"]
            
            # Lookup metadata
            s_meta = self.student_map.get(sid, {"years_of_experience": 0, "college_id": "unknown"})
            j_meta = self.job_map.get(jid, {"seniority_level": 0})
            
            s_skills = self.student_skills_map.get(sid, {})
            j_skills = self.job_skills_map.get(jid, {})
            
            # 1. Skill Overlap & Gap
            if len(j_skills) == 0:
                overlap_ratio = 1.0 # Job with no requirements -> perfect overlap
                prof_gap = 0.0
            else:
                matched_count = 0
                gap_sum = 0.0
                for req_skill, min_prof in j_skills.items():
                    if req_skill in s_skills:
                        actual_prof = s_skills[req_skill]
                        if pd.isnull(actual_prof):
                            # Missing proficiency score treated as 0
                            actual_prof = 0
                        if actual_prof >= min_prof:
                            matched_count += 1
                        else:
                            gap_sum += (min_prof - actual_prof)
                    else:
                        gap_sum += min_prof # Missing skill is a big gap
                
                overlap_ratio = matched_count / len(j_skills)
                prof_gap = (gap_sum / len(j_skills)) / 100.0 # Normalized proficiency gap per required skill
                
            # 2. Experience Fit
            exp_diff = s_meta["years_of_experience"] - j_meta["seniority_level"]
            exp_fit = 1.0 / (1.0 + abs(exp_diff))
            
            # 3. College Prior
            college_id = s_meta["college_id"]
            college_prior = self.priors.get("college_hire_rate", {}).get(college_id, self.priors.get("global_hire_rate", 0.0))
            
            features.append({
                "skill_overlap_ratio": overlap_ratio,
                "proficiency_gap": prof_gap,
                "experience_fit": exp_fit,
                "college_hire_prior": college_prior,
                "global_hire_rate": self.priors.get("global_hire_rate", 0.0), # for explainability
                "college_id": college_id, # for segmentation
                "seniority_level": j_meta["seniority_level"] # for segmentation
            })
            
        return pd.DataFrame(features)
