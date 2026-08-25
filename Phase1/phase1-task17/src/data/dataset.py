"""
data/dataset.py — loads Task 2's cleaned data restricted to Task 7's
locked feature set.
"""
import json
import logging
import pandas as pd

log = logging.getLogger("src.data")


def load_features_and_target(cfg):
    raw_path = cfg.raw_data_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Configured raw_data_path not found: {raw_path}.")
    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {raw_path} is empty.")
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in {raw_path}.")

    locked_path = cfg.locked_features_path
    if not locked_path.exists():
        raise FileNotFoundError(f"Locked feature set not found: {locked_path}.")
    locked = json.loads(locked_path.read_text())
    features = locked.get("final_feature_set")
    if not features:
        raise ValueError(f"'final_feature_set' missing/empty in {locked_path}.")

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
    if y.isna().any():
        raise ValueError("Target column contains missing values.")
    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    log.info("Loaded %s rows x %s locked features + target.", X.shape[0], X.shape[1])
    return X, y
