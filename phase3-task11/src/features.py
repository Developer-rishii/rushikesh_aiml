"""
features.py
===========
The single place feature vectors get built. Both train_ltr.py (offline)
and serve.py (online) import FEATURE_COLUMNS and build_features() from
HERE ONLY. This is the concrete fix for the #1 pitfall called out in the
study guide: "features computed one way in training and another way at
serving". `position` is intentionally NOT a model feature -- it is a
logging artifact of the OLD heuristic, not a property of the candidate,
and leaking it in would let the model "learn" position instead of
relevance (see reports/position_bias_ablation.md).
"""
import pandas as pd

FEATURE_COLUMNS = [
    "skill_match",
    "experience_match",
    "embedding_sim",
    "recency",
    "past_response_rate",
    "profile_completeness",
]

# Columns that exist in logs but must NEVER be used as serving features.
LEAKAGE_COLUMNS = ["position", "true_relevance", "heuristic_score", "is_randomized_slice"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    if X.isnull().any().any():
        # documented, tested fallback for a serving-time feature-store gap
        X = X.fillna(X.median(numeric_only=True))
    return X


def assert_no_leakage(feature_frame: pd.DataFrame):
    leaked = [c for c in LEAKAGE_COLUMNS if c in feature_frame.columns]
    if leaked:
        raise RuntimeError(f"Train/serve skew guard tripped -- leaked columns in features: {leaked}")
