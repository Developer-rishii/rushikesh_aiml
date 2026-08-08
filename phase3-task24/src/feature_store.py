"""
Feature store layer with an explicit freshness check.
Core concept from the study guide: 'Stale features -- serving on old features
is a silent failure; it needs freshness checks and alarms.'
This module makes staleness a first-class, detectable condition instead of a
silent one.
"""
import time


class StaleFeatureError(Exception):
    pass


class FeatureStore:
    """Wraps a features dataframe and enforces a max-age SLA on read."""

    def __init__(self, features_df, max_age_days=2.0, now_fn=time.time):
        self.df = features_df.set_index(["candidate_id", "job_id"])
        self.max_age_days = max_age_days
        self.now_fn = now_fn
        self.corrupted = False  # chaos hook: simulate a corrupted feature table

    def inject_corruption(self, on=True):
        """Chaos hook -- simulate corrupted training/serving data."""
        self.corrupted = on

    def get_features(self, candidate_id, job_id):
        if self.corrupted:
            # Corrupted store returns NaNs / out-of-range values instead of crashing --
            # this is the realistic failure mode we must detect, not a clean exception.
            return {
                "skill_match": float("nan"),
                "exp_match": float("nan"),
                "location_match": -1,
                "age_days": 0.0,
                "stale": False,
                "corrupted": True,
            }

        try:
            row = self.df.loc[(candidate_id, job_id)]
        except KeyError:
            # cold start -> conservative defaults, explicitly flagged
            return {
                "skill_match": 0.3, "exp_match": 0.3, "location_match": 0,
                "age_days": 0.0, "stale": False, "corrupted": False, "cold_start": True,
            }

        age_days = (self.now_fn() - row["feature_ts"]) / 86400.0
        stale = age_days > self.max_age_days
        return {
            "skill_match": float(row["skill_match"]),
            "exp_match": float(row["exp_match"]),
            "location_match": int(row["location_match"]),
            "age_days": float(age_days),
            "stale": bool(stale),
            "corrupted": False,
        }

    @staticmethod
    def validate(features: dict) -> bool:
        """Data-quality gate. Returns False if features are unsafe to feed the model."""
        if features.get("corrupted"):
            return False
        for k in ("skill_match", "exp_match"):
            v = features.get(k)
            if v is None or v != v:  # NaN check
                return False
            if not (0.0 <= v <= 1.0):
                return False
        if features.get("location_match") not in (0, 1):
            return False
        return True
