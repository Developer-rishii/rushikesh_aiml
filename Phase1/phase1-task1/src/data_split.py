"""
Step 3 of the build pipeline: Make a clean train/validation/test split
(stratified, since this is a classification task).

Guards against the #1 pitfall in the study guide: leaking test data into
training. Test data is written out and never touched again until final
evaluation.
"""
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from configs.config import (
    SEED, RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_split")


def split_data(df: pd.DataFrame):
    if not abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    y = df[TARGET_COL]

    # First carve off test set.
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_FRAC, stratify=y, random_state=SEED
    )

    # Then split remainder into train/val.
    val_relative = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative,
        stratify=train_val_df[TARGET_COL], random_state=SEED,
    )

    # Sanity: no row overlap (guards against leakage silently creeping in).
    idx_overlap = set(train_df.index) & set(val_df.index) & set(test_df.index)
    if idx_overlap:
        raise RuntimeError("Data leakage detected: overlapping indices across splits.")

    log.info(
        "Split sizes -> train: %d, val: %d, test: %d (ratios %.2f/%.2f/%.2f)",
        len(train_df), len(val_df), len(test_df), TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
    )
    return train_df, val_df, test_df


def run():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} missing — run data_ingestion.py first.")
    df = pd.read_csv(RAW_PATH)
    train_df, val_df, test_df = split_data(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    log.info("Wrote train/val/test CSVs to data/")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run()
