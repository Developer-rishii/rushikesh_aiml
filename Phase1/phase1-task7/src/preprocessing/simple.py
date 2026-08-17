"""
Lightweight numeric preprocessing (impute + scale) for the engineered
feature matrix — all engineered + original columns are numeric here, so
this is deliberately simpler than Task 4's full ColumnTransformer.
Still fit-on-train-only, same discipline.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_transform_train(X_train: pd.DataFrame, cfg):
    imputer = SimpleImputer(strategy=cfg.numeric_impute_strategy)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

    scaler = None
    if cfg.scale_numeric:
        scaler = StandardScaler()
        X_train_out = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns, index=X_train.index)
    else:
        X_train_out = X_train_imp
    return X_train_out, imputer, scaler


def transform(X: pd.DataFrame, imputer, scaler) -> pd.DataFrame:
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    if scaler is not None:
        return pd.DataFrame(scaler.transform(X_imp), columns=X.columns, index=X.index)
    return X_imp
