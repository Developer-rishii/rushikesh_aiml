"""
Learning-to-rank via pointwise regression onto graded relevance
(Section 5: "Ranking is not classification... pointwise/pairwise/listwise
objectives optimise the ORDER of results"). Pointwise chosen deliberately
-- see README "Alternative approaches" for what was rejected and why.

Two models are trained on the SAME features/data so quality is comparable:
  - baseline: large GradientBoostingRegressor (the "before" / expensive model)
  - small:    right-sized GradientBoostingRegressor (the "after" model)
"""
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES = [
    "age_norm",
    "Action",
    "Comedy",
    "Drama",
    "SciFi",
    "Romance",
]


def split(df):
    train, test = train_test_split(df, test_size=0.25, random_state=21)
    return train, test


def train_model(train_df, n_estimators, max_depth):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=21
    )
    t0 = time.perf_counter()
    model.fit(train_df[FEATURES], train_df["relevance_grade"])
    train_seconds = time.perf_counter() - t0
    return model, train_seconds


def ndcg_at_k(y_true_by_group, y_score_by_group, k=10):
    """Mean nDCG@k across job groups. y_true/score_by_group: list of arrays."""
    scores = []
    for yt, ys in zip(y_true_by_group, y_score_by_group):
        order = np.argsort(-ys)[:k]
        gains = (2 ** np.array(yt)[order] - 1)
        discounts = 1 / np.log2(np.arange(2, len(order) + 2))
        dcg = np.sum(gains * discounts)
        ideal_order = np.argsort(-np.array(yt))[:k]
        ideal_gains = (2 ** np.array(yt)[ideal_order] - 1)
        idcg = np.sum(ideal_gains * discounts[: len(ideal_order)])
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(y_true_by_group, y_score_by_group, k=10, relevance_threshold=2):
    precisions = []
    for yt, ys in zip(y_true_by_group, y_score_by_group):
        order = np.argsort(-ys)[:k]
        relevant = np.array(yt)[order] >= relevance_threshold
        precisions.append(relevant.mean() if len(order) else 0.0)
    return float(np.mean(precisions)) if precisions else 0.0


def grouped_eval(test_df, scores, group_col="job_id", k=10, max_groups=500):
    """Group predictions by job_id (a ranking = candidates ranked per job)."""
    df = test_df.copy()
    df["_score"] = scores
    groups = list(df.groupby(group_col))[:max_groups]
    y_true, y_score = [], []
    for _, g in groups:
        if len(g) < 2:
            continue
        y_true.append(g["relevance_grade"].values)
        y_score.append(g["_score"].values)
    ndcg = ndcg_at_k(y_true, y_score, k)
    prec = precision_at_k(y_true, y_score, k)
    return dict(ndcg_at_k=round(ndcg, 4), precision_at_k=round(prec, 4), n_groups=len(y_true))


def train_serve_skew_check(model, test_df, n=2000, seed=21):
    """
    Section 5: "features computed one way in training and another way at
    serving... Detect it or your model quietly rots."

    Simulate a serve-time feature path that recomputes embedding_sim with
    float32 rounding + a stale location_match cache (a realistic skew
    source), then measure prediction drift vs the training-time path.
    """
    rng = np.random.default_rng(seed)
    sample = test_df.sample(min(n, len(test_df)), random_state=seed).copy()
    train_path_scores = model.predict(sample[FEATURES])

    serve_sample = sample.copy()
    serve_sample["age_norm"] = serve_sample["age_norm"].astype(np.float32)
    stale_mask = rng.random(len(serve_sample)) < 0.05  # 5% stale cache reads
    serve_sample.loc[stale_mask, "Action"] = 1 - serve_sample.loc[stale_mask, "Action"]
    serve_path_scores = model.predict(serve_sample[FEATURES])

    drift = np.abs(train_path_scores - serve_path_scores)
    return dict(
        mean_abs_drift=float(np.mean(drift)),
        max_abs_drift=float(np.max(drift)),
        pct_rows_stale_feature=float(stale_mask.mean() * 100),
    )
