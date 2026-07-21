"""
Detects train/serve skew by recomputing features for a sample of rows through
the serving-path function and diffing against the training dataframe.

Run twice on purpose: once against the BUGGY serving implementation (to prove
the check actually catches a real bug, not just a rubber stamp), then again
against the real (shared) implementation, to prove the fix. This directly
targets the pitfall "Optimising before profiling" 's sibling pitfall listed
in the guide: silent skew that makes offline metrics lie.
"""
import numpy as np
import pandas as pd
from src.config import INTERACTIONS_CSV, CHEAP_FEATURES
from src.data_pipeline import compute_cheap_features, compute_cheap_features_BUGGY


def check(df: pd.DataFrame, fn, label: str, n_sample=500, tol=1e-6):
    sample = df.sample(n=min(n_sample, len(df)), random_state=1)
    max_abs_diff = {f: 0.0 for f in CHEAP_FEATURES}
    n_mismatched = 0
    for _, row in sample.iterrows():
        served = fn(row.to_dict())
        row_mismatch = False
        for f in CHEAP_FEATURES:
            diff = abs(served[f] - row[f])
            max_abs_diff[f] = max(max_abs_diff[f], diff)
            if diff > tol:
                row_mismatch = True
        if row_mismatch:
            n_mismatched += 1

    passed = n_mismatched == 0
    print(f"[{label}] sampled={len(sample)} mismatched_rows={n_mismatched} "
          f"pass={passed}")
    for f, d in max_abs_diff.items():
        flag = "  <-- SKEW" if d > tol else ""
        print(f"    {f}: max_abs_diff={d:.4f}{flag}")
    return {"label": label, "n_sample": len(sample), "n_mismatched": n_mismatched,
            "pass": passed, "max_abs_diff": max_abs_diff}


def main():
    df = pd.read_csv(INTERACTIONS_CSV)
    print("=== Skew check against BUGGY serving implementation ===")
    buggy_result = check(df, compute_cheap_features_BUGGY, "buggy-serving")
    print("\n=== Skew check against REAL (shared) serving implementation ===")
    fixed_result = check(df, compute_cheap_features, "shared-serving")
    assert buggy_result["pass"] is False, "sanity: the buggy path should be caught"
    assert fixed_result["pass"] is True, "sanity: the shared path must be clean"
    print("\nResult: skew check correctly FAILS the buggy path and PASSES the "
          "shared-implementation path used in real serving.")
    return buggy_result, fixed_result


if __name__ == "__main__":
    main()
