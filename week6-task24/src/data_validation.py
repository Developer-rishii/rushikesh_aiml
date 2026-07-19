"""
Data validation & cleaning.

Real data is never clean. This module is the "does the pipeline survive
contact with messy input" layer: schema checks, missing-value handling,
duplicate collapsing, and outlier capping -- each logged so a reviewer can
see exactly what was done to the data before it reached the model.
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "candidate_id", "job_id", "college_tier", "skill_score", "years_exp",
    "jd_match", "portfolio_score", "historical_recommended",
}


class DataValidationError(Exception):
    """Raised when input data cannot be safely used -- missing columns,
    empty file, or a protected-attribute group that has vanished entirely."""


def validate_schema(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        raise DataValidationError("Input dataframe is empty -- nothing to audit or train on.")
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Input data is missing required columns: {sorted(missing_cols)}")
    if df["college_tier"].nunique() < 2:
        raise DataValidationError(
            "Protected attribute 'college_tier' has fewer than 2 groups present -- "
            "cannot compute disparate impact."
        )


def clean(df: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Returns a cleaned copy. If `report` (a dict) is passed, mutates it
    in-place with a log of what was fixed, for the live-demo evidence trail."""
    validate_schema(df)
    df = df.copy()
    log = report if report is not None else {}

    n_before = len(df)
    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log["duplicates_removed"] = n_dupes

    missing_before = {c: int(df[c].isna().sum()) for c in ["jd_match", "portfolio_score"]}
    for col in ["jd_match", "portfolio_score"]:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
    log["missing_values_imputed_with_median"] = missing_before

    # Outlier capping: years_exp beyond a sane career length (winsorize at 99th pct)
    cap = df["years_exp"].quantile(0.99)
    n_capped = int((df["years_exp"] > cap).sum())
    df["years_exp"] = np.minimum(df["years_exp"], cap)
    log["years_exp_outliers_capped"] = {"cap_value": float(cap), "rows_capped": n_capped}

    log["rows_before"] = n_before
    log["rows_after"] = len(df)
    return df
