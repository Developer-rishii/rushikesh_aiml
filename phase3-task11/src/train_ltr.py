"""
train_ltr.py
============
Trains the Learning-to-Rank models on logged impressions + outcomes.

DESIGN DECISION (written down, as the study guide requires):
  We wanted GBDT LambdaMART (LightGBM/XGBoost, the industry-standard
  pairwise/listwise objective). This sandbox has no network access and
  neither library is installed, so LambdaMART is NOT available here.
  REJECTED alternative: approximate a GBDT pairwise ranker by scoring
  each item against a zero-reference vector -- rejected because that
  scoring trick is only valid for models linear in the input, which
  defeats the point of using trees.
  CHOSEN: linear pairwise ranking (RankSVM/RankNet-style logistic
  regression over feature differences) as the primary model -- this is
  a real, production-proven LTR family (used at Bing/early LTR systems),
  and its score = w . x is exactly correct to apply at serving time.
  ALSO BUILT for comparison (the "alternative approach" the guide asks
  us to consider and reject-or-keep with evidence): a listwise linear
  ranker (ListNet-style top-1 softmax cross-entropy).
  Production recommendation, stated explicitly in reports/design_decision.md:
  re-fit with LightGBM `lambdarank` objective on this same feature/label
  pipeline once infra allows -- nothing else in this pipeline needs to change.

Two labels are compared to demonstrate position-bias correction:
  - RAW label   : outcome_label straight from logs (confounded by position)
  - CORRECTED   : outcome_label * inverse-propensity-weight(position)
                  (position itself is EXCLUDED from features either way --
                  see features.py -- corrected model additionally trains on
                  de-biased relevance targets, not just clean features)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from features import FEATURE_COLUMNS, build_features, assert_no_leakage
from position_bias import estimate_propensities, ips_weight

RNG = np.random.default_rng(7)
MAX_PAIRS_PER_JOB = 60


def _build_pairs(df, label_col, group_col="job_id"):
    """Sample preference pairs within each job group for pairwise training."""
    diffs, targets, weights = [], [], []
    for _, g in df.groupby(group_col):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(idx_pairs) > MAX_PAIRS_PER_JOB:
            sel = RNG.choice(len(idx_pairs), MAX_PAIRS_PER_JOB, replace=False)
            idx_pairs = [idx_pairs[s] for s in sel]
        X = g[FEATURE_COLUMNS].values
        labels = g[label_col].values
        for i, j in idx_pairs:
            if labels[i] == labels[j]:
                continue
            sign = 1 if labels[i] > labels[j] else -1
            diffs.append(X[i] - X[j])
            targets.append(sign)
            weights.append(abs(labels[i] - labels[j]))
    return np.array(diffs), np.array(targets), np.array(weights)


def train_pairwise_linear(df, label_col):
    diffs, targets, weights = _build_pairs(df, label_col)
    clf = LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0)
    clf.fit(diffs, targets, sample_weight=weights)
    w = clf.coef_[0]
    return w  # score(x) = w . x


def train_listwise_linear(df, label_col, group_col="job_id", lr=0.05, epochs=150):
    """ListNet-style top-1 softmax cross entropy, linear scorer, plain numpy GD."""
    w = np.zeros(len(FEATURE_COLUMNS))
    groups = list(df.groupby(group_col))
    for _ in range(epochs):
        grad = np.zeros_like(w)
        for _, g in groups:
            X = g[FEATURE_COLUMNS].values
            y = g[label_col].values.astype(float)
            if y.sum() <= 0:
                continue
            p_true = y / y.sum()
            scores = X @ w
            scores = scores - scores.max()
            p_pred = np.exp(scores) / np.exp(scores).sum()
            grad += X.T @ (p_pred - p_true)
        w -= lr * grad / len(groups)
    return w


def score_with_weights(feature_frame: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    assert_no_leakage(feature_frame)
    return feature_frame[FEATURE_COLUMNS].values @ w


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RAW and CORRECTED label columns to the (train-split) log frame."""
    props, eta = estimate_propensities(df)
    df = df.copy()
    df["label_raw"] = df["outcome_label"].astype(float)
    df["ips_w"] = df["position"].apply(lambda p: ips_weight(p, props))
    df["label_corrected"] = df["outcome_label"] * df["ips_w"]
    return df, props, eta


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "raw_logs.csv")
    df, props, eta = prepare_labels(df)
    w_raw = train_pairwise_linear(df, "label_raw")
    w_corr = train_pairwise_linear(df, "label_corrected")
    print("Pairwise weights (RAW, position-confounded):    ", np.round(w_raw, 3))
    print("Pairwise weights (IPS-CORRECTED):                ", np.round(w_corr, 3))
