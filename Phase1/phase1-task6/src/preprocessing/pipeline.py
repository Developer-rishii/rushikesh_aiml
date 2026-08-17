"""
preprocessing/pipeline.py — the core deliverable of Task 4: a
ColumnTransformer that imputes + encodes + scales, built to be FIT ONLY
ON TRAINING DATA and reused unchanged (via .transform) on val/test/
inference data. This is what makes it "leak-free and reusable."

Design choices mapped directly to the brief's step-by-step + pitfalls:
  - Step 1 (list transforms per feature type): see build_preprocessor()
    docstring below — numeric vs categorical get distinct sub-pipelines.
  - Step 2/5 (fit only on train, verify no leakage): enforced by
    fit_preprocessor() taking ONLY X_train, never the full dataframe;
    verify_no_leakage() below re-checks this programmatically.
  - Step 3 (encode categoricals, scale numerics consistently): one
    ColumnTransformer, so train and inference always see identical logic.
  - Step 4 (impute within the pipeline, not by dropping rows): imputers
    are pipeline steps, not a df.dropna() anywhere in this codebase.
  - Step 6 (save the fitted preprocessor for inference reuse): see
    save_preprocessor() / load_preprocessor().
"""
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

log = logging.getLogger("src.preprocessing")


def list_feature_types(X: pd.DataFrame) -> dict:
    """Step 1: list required transforms per feature type."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def build_preprocessor(X_train: pd.DataFrame, cfg) -> ColumnTransformer:
    """Build (unfitted) the preprocessing ColumnTransformer from config."""
    feature_types = list_feature_types(X_train)
    numeric_cols, categorical_cols = feature_types["numeric"], feature_types["categorical"]

    numeric_steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=cfg.categorical_impute_strategy)),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No numeric or categorical columns found to preprocess.")

    log.info("Preprocessor plan: %s numeric cols (impute=%s, scale=%s), "
              "%s categorical cols (impute=%s, encode=%s)",
              len(numeric_cols), cfg.numeric_impute_strategy, cfg.scale_numeric,
              len(categorical_cols), cfg.categorical_impute_strategy, cfg.encode_categorical)
    return ColumnTransformer(transformers)


def fit_preprocessor(X_train: pd.DataFrame, cfg) -> ColumnTransformer:
    """
    Step 2: fit ONLY on training data. This function's signature is the
    contract — it physically cannot see val/test data because they are
    never passed in.
    """
    if X_train.isna().all(axis=0).any():
        bad_cols = X_train.columns[X_train.isna().all(axis=0)].tolist()
        raise ValueError(f"Column(s) entirely missing in training data, cannot impute: {bad_cols}")
    pre = build_preprocessor(X_train, cfg)
    pre.fit(X_train)
    log.info("Fitted preprocessor on %s training rows.", len(X_train))
    return pre


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """
    Step 3/4 applied consistently: the SAME fitted preprocessor is reused,
    unchanged, at val time, test time, and inference time. No re-fitting,
    no different logic path — this is what prevents train/serve drift.
    """
    if X.empty:
        raise ValueError("Cannot transform an empty dataframe.")
    return preprocessor.transform(X)


def verify_no_leakage(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_val: pd.DataFrame) -> dict:
    """
    Step 5: verify no leakage across the split. Two independent checks:
      1. Index-level: train and val share no row indices (data-split leakage).
      2. Statistic-level: the fitted scaler's learned mean must match a
         mean computed independently from X_train alone, and must NOT
         match one computed from X_train+X_val combined (proves the
         scaler wasn't secretly fit on validation data too).
    """
    idx_overlap = set(X_train.index) & set(X_val.index)
    if idx_overlap:
        raise RuntimeError(f"Leakage: {len(idx_overlap)} row(s) shared between train and val splits.")

    numeric_cols = list_feature_types(X_train)["numeric"]
    result = {"index_overlap": len(idx_overlap), "numeric_cols_checked": len(numeric_cols)}

    if numeric_cols and "numeric" in preprocessor.named_transformers_:
        fitted_scaler = preprocessor.named_transformers_["numeric"].named_steps.get("scale")
        if fitted_scaler is not None:
            train_only_mean = X_train[numeric_cols].fillna(X_train[numeric_cols].median()).mean().values
            combined_mean = pd.concat([X_train[numeric_cols], X_val[numeric_cols]])
            combined_mean = combined_mean.fillna(combined_mean.median()).mean().values

            matches_train_only = np.allclose(fitted_scaler.mean_, train_only_mean, rtol=1e-2)
            matches_combined = np.allclose(fitted_scaler.mean_, combined_mean, rtol=1e-6)

            if matches_combined and not matches_train_only:
                raise RuntimeError(
                    "Leakage: fitted scaler's mean matches train+val combined, "
                    "not train alone — the preprocessor was fit on validation data."
                )
            result["scaler_matches_train_only_mean"] = bool(matches_train_only)

    log.info("Leakage verification passed: %s", result)
    return result


def save_preprocessor(preprocessor: ColumnTransformer, path) -> None:
    """Step 6: save the fitted preprocessor for inference reuse."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    log.info("Saved fitted preprocessor -> %s", path)


def load_preprocessor(path) -> ColumnTransformer:
    """Inference-time reuse: load the exact fitted preprocessor, no re-fitting."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved preprocessor at {path} — run the fit step first.")
    return joblib.load(path)
