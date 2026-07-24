"""
Stage B.2 - Build ranking models on real logged data (not curated sample).

baseline_v1: what PlaceMux already has in production (fewer features,
             linear-ish, pointwise).
candidate_v2: the challenger model this task exists to safely test.

Both are pointwise regressors predicting relevance, then used to RANK
candidates for a job (learning-to-rank via regression scores - documented
in the study guide's "role foundations" as a valid pointwise approach).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES_BASELINE = ["skill_match", "location_match"]
FEATURES_CANDIDATE = ["skill_match", "exp_gap", "location_match", "recency"]


@dataclass
class TrainedModel:
    name: str
    version: str
    estimator: object
    features: list

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].values
        return self.estimator.predict(X)


def train_test_split_logs(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Held-out split the model was NOT tuned on (Stage B.3 requirement)."""
    return train_test_split(df, test_size=test_frac, random_state=seed, shuffle=True)


def train_baseline(train_df: pd.DataFrame) -> TrainedModel:
    est = Ridge(alpha=1.0, random_state=42)
    est.fit(train_df[FEATURES_BASELINE].values, train_df["true_relevance"].values)
    return TrainedModel("baseline", "v1", est, FEATURES_BASELINE)


def train_candidate(train_df: pd.DataFrame) -> TrainedModel:
    est = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
    )
    est.fit(train_df[FEATURES_CANDIDATE].values, train_df["true_relevance"].values)
    return TrainedModel("candidate", "v2", est, FEATURES_CANDIDATE)
