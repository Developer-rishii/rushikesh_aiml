"""
Step 5 of the build pipeline: check class balance and base rates.

Run standalone: python src/balance_report.py
Writes: outputs/reports/balance_report.md
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from configs.config import CLEAN_PATH, REPORTS_DIR, TARGET_COL


def run_balance_report(df: pd.DataFrame, target: str):
    counts = df[target].value_counts().sort_index()
    rates = df[target].value_counts(normalize=True).sort_index()
    majority_rate = rates.max()
    return counts, rates, majority_rate


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    counts, rates, majority_rate = run_balance_report(df, TARGET_COL)

    lines = ["# Class Balance Report\n"]
    lines.append("| class (0=malignant, 1=benign) | count | rate |")
    lines.append("|---|---|---|")
    for cls in counts.index:
        lines.append(f"| {cls} | {counts[cls]} | {rates[cls]:.2%} |")
    lines.append(f"\n**Majority-class baseline accuracy: {majority_rate:.2%}**")
    lines.append(
        "\nThe classes are moderately imbalanced (~63/37), not extreme, but "
        f"a majority-class baseline already scores {majority_rate:.1%} "
        "accuracy, so accuracy alone is a weak signal of real skill here. "
        "PR-AUC (see config.py SUCCESS_METRIC) is tracked alongside it."
    )
    imbalance_ratio = counts.max() / counts.min()
    lines.append(f"\nImbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        lines.append("Ratio exceeds 3:1 -> class weighting recommended.")
    else:
        lines.append("Ratio is mild -> stratified split is sufficient, "
                      "class weighting optional (still applied for margin of safety).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "balance_report.md"
    report_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved -> {report_path}")
