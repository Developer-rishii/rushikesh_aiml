"""
data module — Step 1 of the architecture: load the dataset and produce a
reproducible, config-driven train/val/test split.

Nothing here hardcodes a path or a fraction; all of it comes from the
Config object built in configs/loader.py.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger("src.data")


def load_dataframe(cfg) -> pd.DataFrame:
    """Load the raw CSV named in config; fail loudly and specifically."""
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


def split_dataframe(df: pd.DataFrame, cfg):
    """Stratified train/val/test split using config fractions + seed."""
    fracs = (cfg.train_frac, cfg.val_frac, cfg.test_frac)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {fracs}")

    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(cfg.val_frac + cfg.test_frac),
        stratify=y, random_state=cfg.seed,
    )
    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=cfg.seed,
    )

    # leakage guard: no shared indices between splits
    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise RuntimeError("Index overlap detected between splits — leakage guard tripped.")

    log.info("Split (seed=%s): train=%s val=%s test=%s",
              cfg.seed, len(X_train), len(X_val), len(X_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
