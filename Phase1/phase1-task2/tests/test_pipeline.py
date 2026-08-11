"""
Smoke test: proves the pipeline runs on the real data carried over from
Task 1, and handles edge cases gracefully.
Run: python tests/test_pipeline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_ingestion import load_task1_raw, enrich_with_records_system_fields
from feature_profiling import profile_features
from leakage_check import run_leakage_check
from balance_report import run_balance_report
from configs.config import TARGET_COL, SEED


def test_full_run_no_crash():
    from run_pipeline import main
    result = main()
    assert result["go_no_go"] in ("GO", "NO-GO (needs iteration)")
    assert result["metrics"]["val_pr_auc"] > 0
    print("PASS: full pipeline smoke test")


def test_leakage_actually_removed():
    raw = load_task1_raw()
    enriched = enrich_with_records_system_fields(raw, seed=SEED)
    clean_df, _, dropped = run_leakage_check(enriched, TARGET_COL)
    assert "pathologist_diagnosis_code" not in clean_df.columns
    assert "patient_record_id" not in clean_df.columns
    assert clean_df.shape[1] - 1 == raw.shape[1] - 1  # back to the original 30 features
    print("PASS: known leaky/id columns removed, original 30 real features preserved")


def test_balance_report_edge_case_single_class():
    raw = load_task1_raw()
    single_class = raw[raw[TARGET_COL] == 0].copy()
    counts, rates, majority_rate = run_balance_report(single_class, TARGET_COL)
    assert majority_rate == 1.0
    print("PASS: balance report handles single-class edge case")


def test_ingestion_missing_file_raises():
    import data_ingestion
    from configs import config as cfg
    original = cfg.RAW_PATH
    try:
        cfg.RAW_PATH = Path("/tmp/does_not_exist_12345.csv")
        data_ingestion.RAW_PATH = cfg.RAW_PATH
        try:
            data_ingestion.load_task1_raw()
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised, "Expected FileNotFoundError for missing raw file"
        print("PASS: missing-file edge case raises clearly, not a silent failure")
    finally:
        cfg.RAW_PATH = original
        data_ingestion.RAW_PATH = original


def test_profiling_handles_real_data_types():
    raw = load_task1_raw()
    enriched = enrich_with_records_system_fields(raw, seed=SEED)
    report = profile_features(enriched, TARGET_COL)
    assert "missing_pct" in report.columns
    leak_row = report[report["feature"] == "pathologist_diagnosis_code"]
    assert not leak_row.empty
    print("PASS: profiling covers real + enriched feature types")


if __name__ == "__main__":
    test_leakage_actually_removed()
    test_balance_report_edge_case_single_class()
    test_ingestion_missing_file_raises()
    test_profiling_handles_real_data_types()
    test_full_run_no_crash()
    print("\nALL TESTS PASSED")
