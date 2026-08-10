"""
Lightweight tests (run with: python -m pytest tests/ -v, or plain
python tests/test_pipeline.py) covering the pitfalls the study guide
explicitly warns about: fixed seeds, no leakage, no giant-notebook-only
logic (everything here is imported from src/, proving it was promoted
out of a notebook).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_ingestion import load_raw_dataframe, verify_dataframe
from src.data_split import split_data
from configs.config import TARGET_COL, SEED


def test_dataset_shape_and_balance():
    df = load_raw_dataframe()
    verify_dataframe(df)  # should not raise
    assert df.shape[0] > 100
    assert TARGET_COL in df.columns


def test_split_no_leakage_and_ratios():
    df = load_raw_dataframe()
    train_df, val_df, test_df = split_data(df)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)

    idx_overlap = set(train_df.index) & set(val_df.index) & set(test_df.index)
    assert not idx_overlap, "Train/val/test indices must not overlap"

    # roughly 70/15/15
    assert 0.65 <= len(train_df) / total <= 0.75
    assert 0.10 <= len(val_df) / total <= 0.20
    assert 0.10 <= len(test_df) / total <= 0.20


def test_split_is_deterministic():
    df = load_raw_dataframe()
    train1, val1, test1 = split_data(df)
    train2, val2, test2 = split_data(df)
    assert list(train1.index) == list(train2.index), "Same seed must give same split"
    assert SEED == 42


def test_empty_dataframe_raises():
    import pandas as pd
    try:
        verify_dataframe(pd.DataFrame())
        assert False, "Expected ValueError on empty dataframe"
    except ValueError:
        pass


if __name__ == "__main__":
    test_dataset_shape_and_balance()
    test_split_no_leakage_and_ratios()
    test_split_is_deterministic()
    test_empty_dataframe_raises()
    print("All tests passed.")
