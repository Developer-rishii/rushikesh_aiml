"""
train_ranker.py
----------------
Baseline (control):  the ranker already in production — a hand-tuned linear
                      heuristic over skill_match and embedding_sim. This is
                      what we must beat, not a strawman.

Treatment (new model): a learned pointwise-regression ranker (predicts
                      graded relevance, then sorts by predicted score).

Design decision & why (Stage A, item 3 — write down WHY, including what you
rejected): the recommended stack calls for LightGBM/XGBoost LambdaMART
(a listwise/pairwise objective). This environment has no network access and
LightGBM/XGBoost are not installed and cannot be pip-installed offline, so
we use scikit-learn's GradientBoostingRegressor trained pointwise on graded
relevance as the closest available substitute. This is a real, documented
constraint, not a preference — see README "Alternative approaches" for the
explicit tradeoff (pointwise regression is a weaker proxy for the true
ranking objective than listwise LambdaMART; it is expected to under-perform
what a production LightGBM ranker would achieve).

Both models are versioned into artifacts/model_registry.json so any decision
can be traced back to the exact model version that produced it (Pitfall #5
in the study guide: "no model versioning").
"""

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import FEATURE_COLUMNS, compute_training_features

REGISTRY_PATH = "artifacts/model_registry.json"


def baseline_score_fn(cand_df: pd.DataFrame, as_of_day: int) -> np.ndarray:
    """Production heuristic: weighted sum of skill_match and embedding_sim.
    Deliberately does NOT use experience/distance/past_ctr/recency — this is
    what "the baseline you must beat" looks like in practice: simple, already
    shipped, not stupid.
    """
    return 0.6 * cand_df["skill_match"].to_numpy() + 0.4 * cand_df["embedding_sim"].to_numpy()


class TreatmentRanker:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=13
        )
        self.version = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X[FEATURE_COLUMNS], y)
        digest = hashlib.sha256(
            (str(self.model.get_params()) + str(len(X))).encode()
        ).hexdigest()[:12]
        self.version = f"treatment-gbrt-{digest}"
        # store the sign of each feature's relationship with relevance, so
        # explain.py can report directionally-correct reasons (a feature
        # that INCREASES relevance vs one that DECREASES it, e.g. distance)
        self.feature_sign = {
            c: float(np.sign(np.corrcoef(X[c], y)[0, 1])) for c in FEATURE_COLUMNS
        }
        return self

    def score(self, cand_df: pd.DataFrame, as_of_day: int) -> np.ndarray:
        if set(FEATURE_COLUMNS).issubset(cand_df.columns):
            # already feature-computed (e.g. the held-out offline eval table)
            return self.model.predict(cand_df[FEATURE_COLUMNS])
        feats = compute_training_features(
            cand_df.assign(query_id=cand_df.get("query_id", 0)), as_of_day=as_of_day
        )
        return self.model.predict(feats[FEATURE_COLUMNS])


def _log_registry(entries):
    try:
        with open(REGISTRY_PATH) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(entries)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def train(hist_df: pd.DataFrame):
    feats = compute_training_features(hist_df, as_of_day=hist_df["posted_day"].max())
    feats = feats.merge(hist_df[["query_id", "candidate_id", "relevance"]], on=["query_id", "candidate_id"])

    # held-out split BY QUERY (never split rows of the same query across
    # train/test — that would leak ranking context and inflate offline metrics)
    rng = np.random.default_rng(0)
    all_qids = feats["query_id"].unique()
    rng.shuffle(all_qids)
    n_test = int(0.2 * len(all_qids))
    test_qids = set(all_qids[:n_test])
    train_mask = ~feats["query_id"].isin(test_qids)

    train_df, test_df = feats[train_mask], feats[~train_mask]

    treatment = TreatmentRanker().fit(train_df, train_df["relevance"].to_numpy())

    _log_registry(
        [
            dict(
                name="baseline_heuristic",
                version="baseline-v1-static",
                trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                params={"weights": {"skill_match": 0.6, "embedding_sim": 0.4}},
                train_rows=None,
            ),
            dict(
                name="treatment_gbrt_ranker",
                version=treatment.version,
                trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                params=treatment.model.get_params(),
                train_rows=int(len(train_df)),
                test_rows=int(len(test_df)),
                feature_columns=FEATURE_COLUMNS,
            ),
        ]
    )
    return treatment, train_df, test_df


if __name__ == "__main__":
    hist = pd.read_csv("data/historical_logs.csv")
    treatment, train_df, test_df = train(hist)
    test_df.to_csv("data/heldout_test.csv", index=False)
    print(f"trained {treatment.version}; train={len(train_df)} test={len(test_df)} rows")
    print(f"registry written to {REGISTRY_PATH}")
