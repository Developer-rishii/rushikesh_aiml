"""
Lightweight fairness sanity-check, run as part of every eval (not once at
the end as a formality -- that's explicitly flagged as a pitfall).

NOTE ON DATA USED: real hiring-fairness audits compare protected
attributes (gender, caste, religion, disability, etc. under India's
DPDP Act / EEO-style constraints). This synthetic dataset intentionally
does NOT model or encode any protected attribute (no such field exists
in the generator) -- fabricating protected-class labels for a demo would
be worse than not testing at all. Instead this script demonstrates the
MECHANISM (demographic-parity-style comparison of score distributions
across a group) using `cand_region` as a non-sensitive stand-in group
field. In production this same function is pointed at the real
protected-attribute columns behind the appropriate access controls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_pipeline import compute_features_batch


def demographic_parity_gap(df, score_col, group_col="cand_region", top_k_frac=0.2):
    """For each group, what fraction of that group's impressions land in the
    top-k-frac of scores overall? A large spread across groups = potential
    disparate impact and should trigger a fairness review before shipping."""
    threshold = df[score_col].quantile(1 - top_k_frac)
    df = df.copy()
    df["in_top_k"] = df[score_col] >= threshold
    rates = df.groupby(group_col)["in_top_k"].mean()
    return rates, float(rates.max() - rates.min())


def main():
    root = Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "data" / "raw" / "interaction_logs.csv")
    test_df = df[df["day_idx"] >= 24].copy()

    import pickle, json
    version = (root / "artifacts" / "models" / "LATEST_VERSION.txt").read_text().strip()
    with open(root / "artifacts" / "models" / version / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X = compute_features_batch(test_df)
    test_df["model_score"] = model.predict(X)

    rates, gap = demographic_parity_gap(test_df, "model_score", group_col="cand_region")
    print("Top-20%-of-score selection rate by region (proxy group, NOT a protected attribute):")
    print(rates.round(4))
    print(f"\nMax-min gap across groups: {gap:.4f}")
    threshold = 0.10
    print(f"Fairness gap threshold: {threshold} -> "
          f"{'PASS' if gap <= threshold else 'FLAG FOR REVIEW'}")


if __name__ == "__main__":
    main()
