"""
train/build.py — Step 1: bundle the fitted pipeline (preprocess + model)
into ONE artifact. Same single-sklearn-Pipeline discipline as Tasks
8-17: preprocessing steps and the model are steps of the SAME Pipeline
object, so there is no code path where they could be saved separately
and one forgotten — the direct structural guard against "saving model
but not preprocessor."
"""
import json
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
)

log = logging.getLogger("src.train.build")

MODEL_BUILDERS = {"logreg": lambda p: LogisticRegression(**p)}

_METRIC_FUNCS = {
    "pr_auc": lambda y, proba, pred: average_precision_score(y, proba),
    "roc_auc": lambda y, proba, pred: roc_auc_score(y, proba),
    "precision": lambda y, proba, pred: precision_score(y, pred),
    "recall": lambda y, proba, pred: recall_score(y, pred),
    "f1": lambda y, proba, pred: f1_score(y, pred),
    "accuracy": lambda y, proba, pred: accuracy_score(y, pred),
}


def load_data(cfg):
    raw_path = cfg.raw_data_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {raw_path} is empty.")
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    locked_path = cfg.locked_features_path
    if not locked_path.exists():
        raise FileNotFoundError(f"Locked feature set not found: {locked_path}")
    features = json.loads(locked_path.read_text()).get("final_feature_set")
    if not features:
        raise ValueError(f"'final_feature_set' missing/empty in {locked_path}")

    for feat in features:
        if feat not in df.columns and feat.startswith("coeff_variation_"):
            m = feat.replace("coeff_variation_", "").replace("_", " ")
            mean_c, err_c = f"mean {m}", f"{m} error"
            if mean_c in df.columns and err_c in df.columns:
                df[feat] = df[err_c] / (df[mean_c].abs() + 1e-6)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Locked feature(s) missing from raw data: {missing}")

    X = df[features]
    y = df[cfg.target_col]
    if y.isna().any() or X.isna().any().any():
        raise ValueError("Data contains missing values in features or target.")
    log.info("Loaded %s rows x %s locked features.", X.shape[0], X.shape[1])
    return X, y, features


def build_pipeline(cfg) -> Pipeline:
    if cfg.model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{cfg.model_name}'. Available: {list(MODEL_BUILDERS.keys())}")
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODEL_BUILDERS[cfg.model_name](cfg.model_params)))
    return Pipeline(steps)


def train_and_evaluate(cfg):
    X, y, features = load_data(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_frac, stratify=y, random_state=cfg.seed,
    )
    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    metrics = {name: round(float(_METRIC_FUNCS[name](y_test, proba, pred)), 4) for name in cfg.metrics}
    log.info("[Step 1] Fitted single Pipeline (%s), test metrics: %s", [s[0] for s in pipeline.steps], metrics)

    return pipeline, metrics, features, {"train": len(X_train), "test": len(X_test)}, X_test, y_test
