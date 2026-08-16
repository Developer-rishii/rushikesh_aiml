"""
features module — Step 1 of the architecture: turn raw columns into a
model-ready sklearn ColumnTransformer, entirely config-driven (which
columns to drop, whether to scale, how to impute).
"""
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

log = logging.getLogger("src.features")


def apply_drop_columns(X: pd.DataFrame, cfg) -> pd.DataFrame:
    drop = [c for c in cfg.drop_columns if c in X.columns]
    if drop:
        log.info("Dropping configured columns: %s", drop)
    return X.drop(columns=drop)


def build_feature_transformer(X: pd.DataFrame, cfg) -> ColumnTransformer:
    """Build (but don't fit) the preprocessing transformer from config."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_steps = [("impute", SimpleImputer(strategy=cfg.impute_strategy))]
    if cfg.scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No usable numeric or categorical columns found for feature building.")

    log.info("Feature transformer: %s numeric, %s categorical columns",
              len(numeric_cols), len(categorical_cols))
    return ColumnTransformer(transformers)
