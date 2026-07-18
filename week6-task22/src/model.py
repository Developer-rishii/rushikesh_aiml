"""
src/model.py

The trained match-scoring model. Gradient-boosted trees over the
explainable feature space in features.py -- chosen (not logistic
regression) because match quality is non-linear (e.g. a huge skill
overlap can't fully compensate for missing the JD's top-weighted
skill), while staying small enough to explain per-prediction via
feature contributions.
"""
import json
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from .features import FEATURE_NAMES
from .metrics import compute_metrics, best_threshold


class MatchModel:
    def __init__(self, version: str = "v1"):
        self.version = version
        self.clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42
        )
        self.threshold = 0.5
        self.trained_on_rows = 0

    def fit(self, feature_df, min_precision=0.6):
        X = feature_df[FEATURE_NAMES].values
        y = feature_df["good_match"].values
        self.clf.fit(X, y)
        y_score = self.clf.predict_proba(X)[:, 1]
        self.threshold = best_threshold(y, y_score, min_precision=min_precision)
        self.trained_on_rows = len(feature_df)
        return self

    def predict_proba(self, feature_df):
        X = feature_df[FEATURE_NAMES].values
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, feature_df):
        return (self.predict_proba(feature_df) >= self.threshold).astype(int)

    def evaluate(self, feature_df):
        y_true = feature_df["good_match"].values
        y_score = self.predict_proba(feature_df)
        y_pred = (y_score >= self.threshold).astype(int)
        m = compute_metrics(y_true, y_pred, y_score)
        m["threshold"] = self.threshold
        m["model_version"] = self.version
        return m

    def feature_importance(self) -> dict:
        return {name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, self.clf.feature_importances_)}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
