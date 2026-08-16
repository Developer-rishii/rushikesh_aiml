"""
data module — load Task 2/3's cleaned data, optionally enrich it with a
realistic categorical field + injected missingness (so this task's
categorical-encoding and imputation steps have real work to do), and
produce a reproducible stratified split.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger("src.data")


def load_dataframe(cfg) -> pd.DataFrame:
    path = cfg.raw_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Configured data.raw_path not found: {path}. "
            f"Check configs/config.yaml -> data.raw_path."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Configured target_col '{cfg.target_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    if df[cfg.target_col].isna().any():
        raise ValueError(f"Target column '{cfg.target_col}' contains missing values.")
    log.info("Loaded %s rows x %s cols from %s", df.shape[0], df.shape[1], path)
    return df


def enrich_dataframe(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Add a realistic categorical field (tissue sample processing site) and
    inject realistic missingness into a couple of numeric columns, so the
    preprocessing pipeline genuinely has to encode + impute rather than
    doing so vacuously on an already-perfect numeric table.
    """
    out = df.copy()
    rng = np.random.default_rng(cfg.seed)

    if cfg.add_categorical_feature:
        out["sample_processing_site"] = rng.choice(
            ["lab_a", "lab_b", "lab_c"], size=len(out), p=[0.5, 0.3, 0.2]
        )
        log.info("Added categorical feature 'sample_processing_site'")

    if cfg.inject_missing_values and cfg.missing_fraction > 0:
        numeric_cols = out.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != cfg.target_col]
        cols_to_hit = numeric_cols[:3]  # first 3 numeric feature columns
        for col in cols_to_hit:
            mask = rng.random(len(out)) < cfg.missing_fraction
            out.loc[mask, col] = np.nan
        log.info("Injected ~%.0f%% missingness into %s", cfg.missing_fraction * 100, cols_to_hit)

    return out


def split_dataframe(df: pd.DataFrame, cfg):
    fracs = (cfg.train_frac, cfg.val_frac, cfg.test_frac)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {fracs}")

    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(cfg.val_frac + cfg.test_frac), stratify=y, random_state=cfg.seed,
    )
    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=cfg.seed,
    )

    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise RuntimeError("Index overlap detected between splits — leakage guard tripped.")

    log.info("Split (seed=%s): train=%s val=%s test=%s",
              cfg.seed, len(X_train), len(X_val), len(X_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
