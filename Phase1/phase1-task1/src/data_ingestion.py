"""
Step 2 of the build pipeline: Load the dataset and verify shape, types
and class balance.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) — a real, widely used
clinical dataset (569 patients, 30 numeric diagnostic features, binary
malignant/benign target). It ships with scikit-learn (no network needed,
so the pipeline is 100% reproducible offline) but is real measured data,
not a synthetic toy set.
"""
import sys
import logging
import pandas as pd
from sklearn.datasets import load_breast_cancer

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from configs.config import RAW_PATH, TARGET_COL, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_ingestion")


def load_raw_dataframe() -> pd.DataFrame:
    """Load the breast cancer dataset into a single DataFrame with target col."""
    try:
        bunch = load_breast_cancer(as_frame=True)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("Failed to load dataset from sklearn: %s", exc)
        raise RuntimeError("Dataset load failed — check scikit-learn install") from exc

    df = bunch.frame.copy()
    df = df.rename(columns={"target": TARGET_COL})
    return df


def verify_dataframe(df: pd.DataFrame) -> None:
    """Fail loudly (not silently) on shape / type / balance problems."""
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    n_rows, n_cols = df.shape
    if n_rows < 100:
        raise ValueError(f"Dataset suspiciously small for a real run: {n_rows} rows.")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found.")

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        log.warning("Found %d missing values — downstream steps must handle this.", n_missing)

    non_numeric = df.drop(columns=[TARGET_COL]).select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    balance = df[TARGET_COL].value_counts(normalize=True).round(3).to_dict()
    log.info("Shape: %s | dtypes OK | missing: %d | class balance: %s", df.shape, n_missing, balance)

    minority_frac = min(balance.values())
    if minority_frac < 0.05:
        log.warning("Severe class imbalance detected (minority frac=%.3f).", minority_frac)


def ingest() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataframe()
    verify_dataframe(df)
    df.to_csv(RAW_PATH, index=False)
    log.info("Raw data written to %s (%d rows, %d cols)", RAW_PATH, *df.shape)
    return df


if __name__ == "__main__":
    ingest()
