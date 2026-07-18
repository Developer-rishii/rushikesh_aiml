"""
generate_data.py  (v3 — threshold-based recommendations, bias changes selection rate)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent

N_STUDENTS = 400
colleges   = ["Tier1_Mumbai","Tier1_Delhi","Tier2_Pune","Tier2_Nagpur",
              "Tier3_Nashik","Tier3_Aurangabad"]
college_tier = {c: int(c[4]) for c in colleges}
regions    = ["urban","semi_urban","rural"]

college_assign = RNG.choice(colleges, N_STUDENTS, p=[.15,.15,.20,.20,.15,.15])
region_assign  = RNG.choice(regions,  N_STUDENTS, p=[.50,.30,.20])
base_skill     = np.array([4.5-college_tier[c]*0.6 for c in college_assign])
skill_score    = np.clip(base_skill+RNG.normal(0,.5,N_STUDENTS), 0, 5).round(2)
trust_score    = np.clip(.75+RNG.normal(0,.12,N_STUDENTS),0,1).round(3)

students = pd.DataFrame({
    "student_id":           [f"S{i:04d}" for i in range(N_STUDENTS)],
    "college_id":           college_assign,
    "college_tier":         [college_tier[c] for c in college_assign],
    "region":               region_assign,
    "skill_score":          skill_score,
    "ai_trust_score":       trust_score,
    "verified_skill_count": RNG.integers(2,9,N_STUDENTS),
    "years_exposure":       np.clip(RNG.normal(1.5,.8,N_STUDENTS),0,5).round(1),
})

N_JOBS = 30
jobs = pd.DataFrame({
    "job_id":    [f"J{i:03d}" for i in range(N_JOBS)],
    "job_tier":  RNG.choice([1,2,3],N_JOBS,p=[.25,.45,.30]),
    "min_skill": RNG.uniform(2.0,4.5,N_JOBS).round(1),
})

# Match scores: one per (student, job)
rows=[]
for _,stu in students.iterrows():
    for _,job in jobs.iterrows():
        gap=stu["skill_score"]-job["min_skill"]
        ms=float(np.clip(.5+gap*.18+RNG.normal(0,.05),0,1))
        rows.append({
            "student_id":           stu["student_id"],
            "college_id":           stu["college_id"],
            "college_tier":         int(stu["college_tier"]),
            "region":               stu["region"],
            "job_id":               job["job_id"],
            "match_score":          round(ms,4),
            "ai_trust_score":       float(stu["ai_trust_score"]),
            "skill_score":          float(stu["skill_score"]),
            "verified_skill_count": int(stu["verified_skill_count"]),
            "years_exposure":       float(stu["years_exposure"]),
        })

df=pd.DataFrame(rows)

# Bias penalty — makes biased score lower for Tier3/rural
tier_pen   = (df["college_tier"]==3).astype(float)*0.22
region_pen = (df["region"]=="rural").astype(float)*0.10
noise      = pd.Series(RNG.uniform(0.6,1.0,len(df)), index=df.index)
df["match_score_biased"] = np.clip(
    df["match_score"]-(tier_pen+region_pen)*noise, 0, 1
).round(4)

# Threshold-based selection: recommended if score > 0.62 (skill-only)
THRESHOLD = 0.62
df["recommended"]            = (df["match_score"]        >= THRESHOLD).astype(int)
df["production_recommended"] = (df["match_score_biased"] >= THRESHOLD).astype(int)

# Admin-reviewed labels
mask    = RNG.random(len(df)) < 0.20
labeled = df[mask].copy()
labeled["is_biased_outcome"]      = (labeled["recommended"]!=labeled["production_recommended"]).astype(int)
labeled["skill_only_recommended"] = labeled["recommended"]
labeled=labeled[["student_id","job_id","college_tier","region",
                  "match_score","match_score_biased",
                  "skill_only_recommended","production_recommended","is_biased_outcome"]]

df.to_csv(OUT/"recommendations.csv", index=False)
students.to_csv(OUT/"students.csv", index=False)
jobs.to_csv(OUT/"jobs.csv", index=False)
labeled.to_csv(OUT/"fairness_labels.csv", index=False)

print(f"Total pairs:   {len(df):,}")
print(f"Labeled:       {len(labeled):,}  ({labeled['is_biased_outcome'].mean():.1%} biased)")
print("\nProduction rec rate by college_tier:")
print(df.groupby("college_tier")["production_recommended"].mean().round(4))
print("\nSkill-only rec rate by college_tier:")
print(df.groupby("college_tier")["recommended"].mean().round(4))
print("\nProduction rec rate by region:")
print(df.groupby("region")["production_recommended"].mean().round(4))
