"""
generate_data.py
Generates synthetic but real-shaped data for Task 01 Phase 3.

Produces:
  data/interaction_logs.csv    — impressions, clicks, applications (online signals)
  data/prediction_logs.csv     — every served score with features + model_version
  data/training_features.csv   — features as computed at training time
  data/serving_features.csv    — features as computed at serving time (with skew injected)
  data/defect_labels.csv       — admin-reviewed defect labels for ML training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent

# ── Config ─────────────────────────────────────────────────────────────────────
N_STUDENTS    = 300
N_JOBS        = 50
N_IMPRESSIONS = 8000     # total recommendation impressions logged
MODEL_VERSIONS = ["v1.0", "v1.1", "v1.2"]

# ── 1. Students & jobs ─────────────────────────────────────────────────────────
students = pd.DataFrame({
    "student_id":         [f"S{i:04d}" for i in range(N_STUDENTS)],
    "college_tier":       RNG.choice([1, 2, 3], N_STUDENTS, p=[0.25, 0.45, 0.30]),
    "region":             RNG.choice(["urban", "semi_urban", "rural"], N_STUDENTS,
                                      p=[0.50, 0.30, 0.20]),
    "skill_score":        np.clip(RNG.normal(3.0, 0.8, N_STUDENTS), 0, 5).round(2),
    "verified_skills":    RNG.integers(2, 9, N_STUDENTS),
    "years_experience":   np.clip(RNG.normal(1.5, 0.8, N_STUDENTS), 0, 5).round(1),
})

jobs = pd.DataFrame({
    "job_id":         [f"J{i:03d}" for i in range(N_JOBS)],
    "required_skill": np.clip(RNG.normal(3.0, 0.7, N_JOBS), 1, 5).round(1),
    "seniority":      RNG.choice(["junior", "mid", "senior"], N_JOBS, p=[0.40, 0.40, 0.20]),
    "sector":         RNG.choice(["tech", "finance", "ops", "data"], N_JOBS),
})

# ── 2. Prediction logs (every served recommendation) ─────────────────────────
base_date = datetime(2026, 4, 1)
rows = []
for _ in range(N_IMPRESSIONS):
    stu   = students.sample(1, random_state=int(RNG.integers(0, 2**31))).iloc[0]
    job   = jobs.sample(1, random_state=int(RNG.integers(0, 2**31))).iloc[0]
    mv    = RNG.choice(MODEL_VERSIONS, p=[0.20, 0.35, 0.45])
    ts    = base_date + timedelta(
                days=int(RNG.integers(0, 90)),
                hours=int(RNG.integers(0, 24)))

    # Offline score: what the model predicted
    gap         = stu["skill_score"] - job["required_skill"]
    offline_score = float(np.clip(0.5 + gap * 0.18 + RNG.normal(0, 0.05), 0, 1))

    # Train/serve skew: skill_score computed slightly differently at serving time
    # (simulates a feature pipeline bug introduced in v1.1)
    skew = 0.0
    if mv in ["v1.1", "v1.2"]:
        skew = float(RNG.normal(0.08, 0.03))   # systematic upward bias

    served_score = float(np.clip(offline_score + skew, 0, 1))

    rows.append({
        "log_id":        f"L{len(rows):06d}",
        "timestamp":     ts.isoformat(),
        "student_id":    stu["student_id"],
        "job_id":        job["job_id"],
        "model_version": mv,
        "rank_position": int(RNG.integers(1, 6)),
        "offline_score": round(offline_score, 4),
        "served_score":  round(served_score, 4),
        "skew":          round(skew, 4),
        # Features at serving time
        "feat_skill_score":     round(stu["skill_score"] + skew * 0.5, 3),
        "feat_skill_gap":       round(gap, 3),
        "feat_verified_skills": int(stu["verified_skills"]),
        "feat_years_exp":       float(stu["years_experience"]),
        "college_tier":         int(stu["college_tier"]),
        "region":               stu["region"],
    })

pred_logs = pd.DataFrame(rows)

# ── 3. Interaction logs (online signals) ──────────────────────────────────────
# CTR: probability of click depends on rank and actual skill match
# Clicks are lower than the offline score would predict — the online/offline gap
int_rows = []
for _, pred in pred_logs.iterrows():
    # Online CTR inversely penalised by rank and train/serve skew
    true_relevance  = pred["offline_score"]
    rank_decay      = 1.0 / np.log2(pred["rank_position"] + 1)
    skew_penalty    = abs(pred["skew"]) * 0.4
    click_prob      = float(np.clip(true_relevance * rank_decay - skew_penalty, 0, 1))

    clicked     = int(RNG.random() < click_prob)
    applied     = int(clicked and RNG.random() < 0.30)
    shortlisted = int(applied and RNG.random() < 0.25)

    int_rows.append({
        "log_id":        pred["log_id"],
        "student_id":    pred["student_id"],
        "job_id":        pred["job_id"],
        "model_version": pred["model_version"],
        "rank_position": pred["rank_position"],
        "timestamp":     pred["timestamp"],
        "clicked":       clicked,
        "applied":       applied,
        "shortlisted":   shortlisted,
        "college_tier":  pred["college_tier"],
        "region":        pred["region"],
    })

interaction_logs = pd.DataFrame(int_rows)

# ── 4. Training features (features as computed at TRAINING time) ──────────────
# At training time, skill_score was computed from raw scores (no skew)
train_feats = pred_logs[["log_id", "student_id", "job_id",
                           "offline_score", "feat_skill_gap",
                           "feat_verified_skills", "feat_years_exp"]].copy()
train_feats.rename(columns={"feat_skill_gap": "skill_gap",
                              "feat_verified_skills": "verified_skills",
                              "feat_years_exp": "years_exp",
                              "offline_score": "match_score_train"}, inplace=True)

# ── 5. Serving features (features at SERVING time, with skew injected) ────────
serving_feats = pred_logs[["log_id", "student_id", "job_id",
                             "served_score", "feat_skill_score",
                             "feat_skill_gap", "feat_verified_skills",
                             "feat_years_exp", "skew"]].copy()
serving_feats.rename(columns={"feat_skill_gap": "skill_gap",
                                "feat_verified_skills": "verified_skills",
                                "feat_years_exp": "years_exp",
                                "served_score": "match_score_served",
                                "feat_skill_score": "skill_score_served"}, inplace=True)

# ── 6. Defect labels (admin-reviewed — ~15% of logs reviewed) ─────────────────
# A defect = the model ranked a clearly irrelevant job high, or missed a clear match
reviewed_mask  = RNG.random(N_IMPRESSIONS) < 0.15
reviewed_logs  = pred_logs[reviewed_mask].copy().reset_index(drop=True)
int_reviewed   = interaction_logs[reviewed_mask].copy().reset_index(drop=True)

# Label: is_defect = 1 if model gave high score but user didn't click (false positive)
#                  OR model gave low score but the job was actually clicked (false negative)
high_score_no_click = (reviewed_logs["served_score"] > 0.70) & (int_reviewed["clicked"] == 0)
low_score_clicked   = (reviewed_logs["served_score"] < 0.40) & (int_reviewed["clicked"] == 1)
is_defect = (high_score_no_click | low_score_clicked).astype(int)

defect_type = pd.Series(["none"] * len(reviewed_logs))
defect_type[high_score_no_click] = "false_positive"
defect_type[low_score_clicked]   = "false_negative"

# User impact: higher rank + more severe score error = higher impact
user_impact = (
    (1.0 / np.log2(reviewed_logs["rank_position"] + 1)) *
    abs(reviewed_logs["served_score"] - reviewed_logs["offline_score"]) * 10
).clip(0, 5).round(3)

defects = pd.DataFrame({
    "log_id":           reviewed_logs["log_id"],
    "student_id":       reviewed_logs["student_id"],
    "job_id":           reviewed_logs["job_id"],
    "model_version":    reviewed_logs["model_version"],
    "served_score":     reviewed_logs["served_score"],
    "offline_score":    reviewed_logs["offline_score"],
    "skew":             reviewed_logs["skew"],
    "rank_position":    reviewed_logs["rank_position"],
    "clicked":          int_reviewed["clicked"],
    "applied":          int_reviewed["applied"],
    "is_defect":        is_defect,
    "defect_type":      defect_type.values,
    "user_impact_score":user_impact.values,
    "college_tier":     reviewed_logs["college_tier"],
    "region":           reviewed_logs["region"],
})

# ── 7. Save all ────────────────────────────────────────────────────────────────
pred_logs.to_csv(OUT / "prediction_logs.csv", index=False)
interaction_logs.to_csv(OUT / "interaction_logs.csv", index=False)
train_feats.to_csv(OUT / "training_features.csv", index=False)
serving_feats.to_csv(OUT / "serving_features.csv", index=False)
defects.to_csv(OUT / "defect_labels.csv", index=False)
students.to_csv(OUT / "students.csv", index=False)
jobs.to_csv(OUT / "jobs.csv", index=False)

print(f"prediction_logs:    {len(pred_logs):,} rows")
print(f"interaction_logs:   {len(interaction_logs):,} rows")
print(f"defect_labels:      {len(defects):,} reviewed ({defects['is_defect'].mean():.1%} defects)")
print(f"Model versions:     {pred_logs['model_version'].value_counts().to_dict()}")
print(f"Overall CTR:        {interaction_logs['clicked'].mean():.3f}")
print(f"Overall apply rate: {interaction_logs['applied'].mean():.3f}")
print("Data saved.")
