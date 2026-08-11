"""
Step 1-2 of the build pipeline: load the real dataset handed off from
Task 1 (Wisconsin Diagnostic Breast Cancer, 569 rows, 30 real measured
features) and define the target precisely.

The raw WDBC feature set itself is clean (no leakage, no missing values,
no ID column) -- it's a curated benchmark dataset. To make the leakage
hunt and feature-quality audit demonstrable on THIS real data rather than
hypothetical, we enrich it with two columns a real hospital records system
would actually attach, and that a careless pipeline would actually ingest:

  - `patient_record_id`      : a unique identifier -> no generalizable
                                signal, should be dropped.
  - `pathologist_diagnosis_code` : populated by the pathology lab AFTER
                                the diagnosis is finalized -- this is the
                                target restated in another encoding, i.e.
                                textbook leakage. In a real hospital
                                extract this kind of field slips into
                                training data constantly (it's sitting
                                right next to the real features in the
                                same records export).

Run standalone: python src/data_ingestion.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from configs.config import RAW_PATH, ENRICHED_PATH, TARGET_COL, SEED


def load_task1_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Expected Task 1's raw data at {RAW_PATH} — copy "
            f"phase1-task1/data/raw.csv here before running Task 2."
        )
    df = pd.read_csv(RAW_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {RAW_PATH}")
    return df


def enrich_with_records_system_fields(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out.insert(0, "patient_record_id", np.arange(500000, 500000 + len(out)))
    # Leaky field: directly derived from the target (with the label's own
    # vocabulary), exactly like a post-diagnosis system field would be.
    out["pathologist_diagnosis_code"] = np.where(
        out[TARGET_COL] == 1, "BENIGN-CONFIRMED", "MALIGNANT-CONFIRMED"
    )
    return out


if __name__ == "__main__":
    df = load_task1_raw()
    enriched = enrich_with_records_system_fields(df)
    enriched.to_csv(ENRICHED_PATH, index=False)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols from Task 1 raw data.")
    print(f"Enriched -> {enriched.shape[0]} rows x {enriched.shape[1]} cols "
          f"-> {ENRICHED_PATH}")
    print(f"Class balance: {enriched[TARGET_COL].value_counts(normalize=True).round(3).to_dict()}")
