"""
demo/demo_walkthrough.py

Study Guide Stage C.2 / Self-check:
  "Ask them to walk you through one real example end-to-end: this
   student, this job, and why it's a match."

Run: python3 demo/demo_walkthrough.py
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import build_features_row
from src.model import MatchModel
from src.explain import explain_match

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_latest.joblib")


def main():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv")).set_index("student_id")
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv")).set_index("job_id")
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions_2026-06.csv"))

    model = MatchModel.load(MODEL_PATH)

    # Pick one real positive and one real negative example from June (post-retrain, live data)
    pos_row = interactions[interactions["good_match"] == 1].iloc[0]
    neg_row = interactions[interactions["good_match"] == 0].iloc[0]

    for label, row in [("GROUND-TRUTH POSITIVE", pos_row), ("GROUND-TRUTH NEGATIVE", neg_row)]:
        sid, jid = row["student_id"], row["job_id"]
        s_skills = json.loads(students.loc[sid, "skills_json"])
        j_skills = json.loads(jobs.loc[jid, "required_skills_json"])
        years_gap = int(students.loc[sid, "years_experience"]) - int(jobs.loc[jid, "years_required"])

        feat = build_features_row(s_skills, j_skills, years_gap)
        feat_df = pd.DataFrame([{**feat, "good_match": row["good_match"]}])
        score = float(model.predict_proba(feat_df)[0])
        decision = int(score >= model.threshold)

        print("=" * 70)
        print(f"{label}  (actual label in data: {'good match' if row['good_match'] else 'not a match'})")
        print("=" * 70)
        print(f"Student {sid}: skills={s_skills}, years_experience={students.loc[sid,'years_experience']}")
        print(f"Job     {jid}: required={j_skills}, years_required={jobs.loc[jid,'years_required']}")
        print()
        print(explain_match(sid, jid, s_skills, j_skills, feat, score, decision, model))
        print()
        correct = "CORRECT" if decision == row["good_match"] else "MISS"
        print(f"Model verdict matches ground truth: {correct}")
        print()


if __name__ == "__main__":
    main()
