"""
Step 3 of the build pipeline: inventory candidate features and profile
each for quality (type, missingness, cardinality, association w/ target).

Run standalone: python src/feature_profiling.py
Writes: outputs/reports/feature_profile.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from configs.config import ENRICHED_PATH, REPORTS_DIR, TARGET_COL


def profile_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    y = df[target]
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        missing_pct = s.isna().mean() * 100
        n_unique = s.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(s):
            corr = s.corr(y) if s.notna().sum() > 1 else np.nan
            signal = "numeric"
        else:
            try:
                rates = df.groupby(col)[target].mean()
                corr = rates.max() - rates.min()
            except Exception:
                corr = np.nan
            signal = "categorical"

        flags = []
        if missing_pct > 30:
            flags.append("high_missingness")
        if n_unique <= 1:
            flags.append("constant")
        if n_unique == n:
            flags.append("likely_id_no_signal")
        if pd.notna(corr) and abs(corr) > 0.9:
            flags.append("suspiciously_high_assoc_check_leakage")

        rows.append({
            "feature": col,
            "dtype": str(s.dtype),
            "signal_type": signal,
            "missing_pct": round(missing_pct, 2),
            "n_unique": n_unique,
            "assoc_with_target": round(float(corr), 4) if pd.notna(corr) else None,
            "quality_flags": ";".join(flags) if flags else "ok",
        })

    return pd.DataFrame(rows).sort_values(
        "assoc_with_target", key=lambda s: s.abs(), ascending=False
    )


if __name__ == "__main__":
    df = pd.read_csv(ENRICHED_PATH)
    report = profile_features(df, TARGET_COL)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "feature_profile.csv"
    report.to_csv(out_path, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved -> {out_path}")
