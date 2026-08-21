"""
data/dataset.py — Step 2: wire data loading and splitting at the front.
Loads Task 2's cleaned data, restricts to Task 7's locked feature set
(so this task consumes prior tasks' vetted output rather than
re-deriving it), and produces a reproducible stratified split.
"""
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger("src.data")


def load_locked_features(cfg) -> list:
    path = cfg.locked_features_path
    if not path.exists():
        raise FileNotFoundError(
            f"Locked feature set not found at {path} — expected Task 7's "
            f"locked_feature_set.json to be copied into data/."
        )
    payload = json.loads(path.read_text())
    features = payload.get("final_feature_set")
    if not features:
        raise ValueError(f"'final_feature_set' missing or empty in {path}")
    return features


def _recompute_known_engineered_features(df: pd.DataFrame, locked_features: list) -> pd.DataFrame:
    """
    Task 7's locked feature set can include engineered columns (derived
    from raw measurements) that don't exist as literal columns in Task 2's
    output. Rather than treating that as schema drift, this recomputes
    the ONE known formula (coefficient of variation = error / mean,
    exactly as defined in Task 7's src/features/engineer.py) so the
    hand-off between tasks stays honoured. Any engineered feature NOT
    covered by this whitelist still correctly triggers the missing-
    feature error below — this is a deliberate, narrow allowlist, not a
    silent catch-all.
    """
    out = df.copy()
    for feat in locked_features:
        if feat in out.columns or not feat.startswith("coeff_variation_"):
            continue
        measurement = feat.replace("coeff_variation_", "").replace("_", " ")
        mean_c, err_c = f"mean {measurement}", f"{measurement} error"
        if mean_c in out.columns and err_c in out.columns:
            out[feat] = out[err_c] / (out[mean_c].abs() + 1e-6)
    return out


def load_dataframe(cfg) -> pd.DataFrame:
    path = cfg.raw_data_path
    if not path.exists():
        raise FileNotFoundError(f"Configured data.raw_path not found: {path}.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataframe from {path} is empty.")
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in {path}. "
                          f"Available columns: {list(df.columns)}")
    if df[cfg.target_col].isna().any():
        raise ValueError(f"Target column '{cfg.target_col}' contains missing values.")

    locked_features = load_locked_features(cfg)
    df = _recompute_known_engineered_features(df, locked_features)
    missing_features = [f for f in locked_features if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Locked feature set references column(s) not present in the raw data "
            f"(schema drift between Task 7 and this dataset): {missing_features}"
        )

    df = df[locked_features + [cfg.target_col]]
    log.info("Loaded %s rows x %s locked features (+target) from %s",
              df.shape[0], len(locked_features), path)
    return df


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
