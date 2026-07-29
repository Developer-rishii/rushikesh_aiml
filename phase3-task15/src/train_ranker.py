"""
train_ranker.py
================
Trains the production ranker.

DESIGN DECISION (Section 5 "Learning-to-rank" + Section 8 alternatives):
  Objective = LightGBM LambdaMART (listwise), grouped by job_id.
  REJECTED plain pointwise classification (predict P(shortlisted) directly
  with logloss) as the primary model because Section 5 explicitly warns
  "Ranking is not classification" -- pointwise objectives optimise
  calibration, not ORDER, and PlaceMux serves a ranked list per job, so
  what matters is whether the top-k order is right, not the absolute
  probability. We keep SkillMatchBaseline (pointwise-trivial) only as the
  baseline to beat, never as a candidate for production.

Labels: shortlisted (binary) is used as the relevance label. Graded
relevance (0=impression,1=click,2=applied,3=shortlisted) was considered
and rejected for v1 to keep the model card's "what does this predict"
story simple; noted in model_card.py limitations as a future improvement.
"""
import lightgbm as lgb
import pandas as pd

from src.features import compute_features, FEATURE_COLUMNS, feature_schema_hash


def _grouped(df):
    """LightGBM ranker needs contiguous group blocks + group sizes."""
    df = df.sort_values("job_id").reset_index(drop=True)
    X = compute_features(df)
    y = df["shortlisted"].values
    group_sizes = df.groupby("job_id").size().values
    return df, X, y, group_sizes


def train_lambdamart(train_df, valid_df=None, params=None):
    train_df, X_tr, y_tr, g_tr = _grouped(train_df)
    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=FEATURE_COLUMNS)

    valid_sets, valid_names = [train_set], ["train"]
    if valid_df is not None:
        valid_df, X_va, y_va, g_va = _grouped(valid_df)
        valid_set = lgb.Dataset(X_va, label=y_va, group=g_va, reference=train_set)
        valid_sets.append(valid_set)
        valid_names.append("valid")

    default_params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10],
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=20,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=150,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(20, verbose=False)] if valid_df is not None else [],
    )
    return model


class RankerWrapper:
    """Thin wrapper so the same .predict(df) interface works for the
    LightGBM model as for SkillMatchBaseline -- keeps serve.py agnostic
    to which model type is currently in production (needed for rollback
    across model families if that ever happens)."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, df: pd.DataFrame):
        X = compute_features(df)
        return self.booster.predict(X)
