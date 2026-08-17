"""
target_check.py — Step 1: re-confirm the target definition and label
quality before engineering anything on top of it. Cheap but mandatory:
if the target itself is broken, no amount of feature engineering matters.
"""
import logging
import pandas as pd

log = logging.getLogger("src.features.target_check")


def confirm_target(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    n_missing = int(y.isna().sum())
    unique_vals = sorted(y.dropna().unique().tolist())
    is_binary = set(unique_vals) <= {0, 1}
    class_counts = y.value_counts().to_dict()

    result = {
        "target_col": target_col,
        "definition": "0 = malignant, 1 = benign (Wisconsin Diagnostic Breast Cancer)",
        "n_rows": len(df),
        "n_missing_labels": n_missing,
        "unique_values": unique_vals,
        "is_clean_binary": bool(is_binary and n_missing == 0),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
    }
    if not result["is_clean_binary"]:
        raise ValueError(f"Target quality check failed: {result}")

    log.info("[Step 1] Target confirmed: %s", result)
    return result
