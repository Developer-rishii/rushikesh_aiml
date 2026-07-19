"""
Guards the #1 failure mode called out in the study guide: train/serve
skew. Confirms that a row taken straight from the training log and the
same logical entity submitted as a serving-time request produce IDENTICAL
feature vectors, because both paths call src/features/feature_pipeline.py's
compute_features(). Run with: pytest tests/test_train_serve_consistency.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.feature_pipeline import compute_features, compute_features_batch, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def test_single_row_matches_batch_computation():
    df = pd.read_csv(ROOT / "data" / "raw" / "interaction_logs.csv", nrows=50)
    batch_feats = compute_features_batch(df)

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        single_feats = compute_features(row)
        for col in FEATURE_COLUMNS:
            assert abs(single_feats[col] - batch_feats.iloc[i][col]) < 1e-9, (
                f"Row {i} feature '{col}' diverged between single-row (serving-style) "
                f"and batch (training-style) computation -- this IS train/serve skew."
            )


def test_serving_request_shape_produces_same_features_as_training_row():
    """Simulate a serving-time request payload (as sent to POST /predict)
    built from the same underlying entity as a training row, and assert
    the feature vector is bit-identical."""
    df = pd.read_csv(ROOT / "data" / "raw" / "interaction_logs.csv", nrows=5)
    row = df.iloc[0].to_dict()

    serving_payload = {
        "candidate_id": row["candidate_id"],
        "job_id": row["job_id"],
        "cand_experience_yrs": row["cand_experience_yrs"],
        "cand_expected_salary": row["cand_expected_salary"],
        "cand_region": row["cand_region"],
        "cand_activity_score": row["cand_activity_score"],
        "cand_skills": row["cand_skills"],
        "job_min_exp": row["job_min_exp"],
        "job_salary_offered": row["job_salary_offered"],
        "job_region": row["job_region"],
        "job_popularity": row["job_popularity"],
        "job_req_skills": row["job_req_skills"],
    }

    train_feats = compute_features(row)
    serve_feats = compute_features(serving_payload)
    assert train_feats == serve_feats
