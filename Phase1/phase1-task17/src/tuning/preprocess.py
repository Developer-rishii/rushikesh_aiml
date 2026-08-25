"""tuning/preprocess.py — fit-on-train-only impute+scale, shared by every trial."""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_preprocessor(X_train: pd.DataFrame, cfg):
    imputer = SimpleImputer(strategy=cfg.numeric_impute_strategy)
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    scaler = None
    if cfg.scale_numeric:
        scaler = StandardScaler()
        X_out = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_train.columns, index=X_train.index)
    else:
        X_out = X_imp
    return X_out, imputer, scaler


def transform(X: pd.DataFrame, imputer, scaler) -> pd.DataFrame:
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    if scaler is not None:
        return pd.DataFrame(scaler.transform(X_imp), columns=X.columns, index=X.index)
    return X_imp
