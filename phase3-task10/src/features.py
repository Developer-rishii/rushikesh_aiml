"""
features.py
------------
Single source of truth for feature *definitions*, but exposes TWO
computation paths on purpose:

  - compute_training_features(): used when building the offline training set
  - compute_serving_features():  used when features are computed live, at
                                  request/serving time

In real systems these two paths are usually written by different people at
different times (a batch ETL job vs. an online feature service) and they
silently drift apart. Stage 7 (skew_check.py) proves whether that happened
here rather than assuming it didn't.

FEATURE_COLUMNS is the contract both paths must satisfy.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "skill_match",
    "experience_years",
    "distance_km",
    "past_ctr",
    "embedding_sim",
    "recency_days",
]


def compute_training_features(raw: pd.DataFrame, as_of_day: int) -> pd.DataFrame:
    """Batch/offline feature computation used to build the training table.

    recency_days is computed relative to `as_of_day`, using integer day
    arithmetic on the posting's `posted_day` field.
    """
    df = raw.copy()
    df["recency_days"] = (as_of_day - df["posted_day"]).clip(lower=0)
    return df[["query_id", "candidate_id"] + FEATURE_COLUMNS]


def compute_serving_features(raw: pd.DataFrame, as_of_day: int) -> pd.DataFrame:
    """Online/serving feature computation, used at request time.

    BUG (intentional, for the skew check): the serving path computes
    recency using a timestamp that has already been rounded to the start
    of the day by an upstream caching layer, and off-by-one'd by a
    timezone conversion that the training pipeline does not apply. This
    mirrors a very common real-world skew source: two teams, two
    "obviously equivalent" implementations of the same feature.
    """
    df = raw.copy()
    cached_posted_day = df["posted_day"] + 1  # timezone/caching bug
    df["recency_days"] = (as_of_day - cached_posted_day).clip(lower=0)
    return df[["query_id", "candidate_id"] + FEATURE_COLUMNS]
