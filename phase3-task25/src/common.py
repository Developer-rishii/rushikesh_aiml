"""
Shared feature computation layer (Sec 5 'Train/serve skew' + Sec 6
'disciplined feature-computation layer').

CRITICAL: this module is imported by BOTH train_ranker.py and every serving
path (latency_bench, dr_failover, explainability). There is exactly one
definition of each feature, computed the same way in training and serving,
so it is structurally impossible for training and serving to diverge.
"""
import pandas as pd

FEATURES = ["skill_score", "exp_years", "job_seniority", "job_comp_level", "fit_gap"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fit_gap"] = -(out.skill_score * 5 - out.job_seniority).abs()
    return out

def get_X(df: pd.DataFrame):
    df = build_features(df)
    return df[FEATURES]
