"""
data/dataset.py — loads Task 2's cleaned data and splits off the target
column (kept aside, unused during clustering prep, only for the external
sanity-check reference in the final report).
"""
import logging
import pandas as pd

log = logging.getLogger("src.data")


def load_unsupervised_features(cfg):
    path = cfg.raw_data_path
    if not path.exists():
        raise FileNotFoundError(f"Configured data.raw_path not found: {path}.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in {path}.")

    y_reference = df[cfg.target_col].copy()
    X = df.drop(columns=[cfg.target_col])
    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values — clustering requires complete data.")

    log.info("Loaded %s rows x %s candidate features (target held aside, unused in prep)", *X.shape)
    return X, y_reference
