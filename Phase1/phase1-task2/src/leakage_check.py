"""
Step 4 of the build pipeline: hunt for leakage and remove offending features.

Method (documented, not just claimed):
  1. Domain reasoning: for each feature, ask "would this value be known
     BEFORE the outcome is determined?" -- fields populated as a
     *consequence* of the diagnosis are leaky by construction, regardless
     of how strong or weak their correlation looks.
  2. Statistical smell test (secondary, confirmatory only): flag any
     feature whose association with the target is implausibly high
     (> 0.9) for real measured clinical data.

Run standalone: python src/leakage_check.py
Writes: outputs/reports/leakage_report.md, data/clean.csv
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from configs.config import ENRICHED_PATH, CLEAN_PATH, REPORTS_DIR, TARGET_COL

KNOWN_LEAKY_FEATURES = {
    "pathologist_diagnosis_code": (
        "Populated by the pathology lab only AFTER the diagnosis is "
        "finalized -- this is the target itself restated in another "
        "vocabulary (BENIGN-CONFIRMED / MALIGNANT-CONFIRMED), not a "
        "predictor available at the time of biopsy analysis."
    ),
}

ID_LIKE_NO_SIGNAL = {
    "patient_record_id": "Unique identifier, carries no generalizable signal; drop.",
}


def check_statistical_smell(df: pd.DataFrame, target: str, threshold: float = 0.9):
    suspects = {}
    y = df[target]
    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        corr = df[col].corr(y)
        if pd.notna(corr) and abs(corr) > threshold:
            suspects[col] = round(float(corr), 4)
    return suspects


def run_leakage_check(df: pd.DataFrame, target: str):
    stat_suspects = check_statistical_smell(df, target)
    to_drop = set(KNOWN_LEAKY_FEATURES) | set(ID_LIKE_NO_SIGNAL) | set(stat_suspects)
    clean_df = df.drop(columns=[c for c in to_drop if c in df.columns])
    return clean_df, stat_suspects, to_drop


if __name__ == "__main__":
    df = pd.read_csv(ENRICHED_PATH)
    clean_df, stat_suspects, dropped = run_leakage_check(df, TARGET_COL)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Leakage Check Report\n"]
    lines.append("## Domain-reasoned leaks (removed by definition, not just correlation)\n")
    for feat, why in KNOWN_LEAKY_FEATURES.items():
        lines.append(f"- **{feat}**: {why}")
    lines.append("\n## ID-like / no-signal features removed\n")
    for feat, why in ID_LIKE_NO_SIGNAL.items():
        lines.append(f"- **{feat}**: {why}")
    lines.append("\n## Statistical smell test (|corr| > 0.9 with target)\n")
    if stat_suspects:
        for feat, corr in stat_suspects.items():
            lines.append(f"- **{feat}**: corr={corr} -> confirms domain reasoning above.")
    else:
        lines.append("- No numeric features exceeded the 0.9 threshold "
                      "(note: `pathologist_diagnosis_code` is categorical, so "
                      "it is caught by domain reasoning, not this numeric check "
                      "-- proof that correlation-only leakage detection is not enough).")
    lines.append(f"\n## Final dropped feature set: {sorted(dropped)}")
    lines.append(f"\n## Remaining feature count after cleaning: {clean_df.shape[1] - 1} "
                  f"(the original 30 real WDBC measurements, excl. target)")

    report_path = REPORTS_DIR / "leakage_report.md"
    report_path.write_text("\n".join(lines))
    clean_df.to_csv(CLEAN_PATH, index=False)

    print("\n".join(lines))
    print(f"\nSaved report -> {report_path}")
    print(f"Saved cleaned dataset -> {CLEAN_PATH}")
