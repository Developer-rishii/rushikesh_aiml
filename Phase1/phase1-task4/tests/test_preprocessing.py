"""
Tests for Task 4. Each pitfall named in the brief gets its own dedicated
test, plus a live end-to-end run and edge cases.
Run: python tests/test_preprocessing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from configs.loader import load_config
from src.data.dataset import load_dataframe, enrich_dataframe, split_dataframe
from src.preprocessing.pipeline import (
    fit_preprocessor, transform, verify_no_leakage,
    save_preprocessor, load_preprocessor, list_feature_types,
)


def _setup():
    cfg = load_config()
    raw = load_dataframe(cfg)
    enriched = enrich_dataframe(raw, cfg)
    splits = split_dataframe(enriched, cfg)
    return cfg, enriched, splits


def test_live_end_to_end_run():
    from src.run_protocol import main
    result = main()
    assert result["inference_reuse_verified"] is True
    assert result["rows_dropped_during_preprocessing"] == 0
    print("PASS: live end-to-end protocol run, artifact + report produced")


def test_pitfall_scaler_not_fit_on_all_data():
    """Pitfall: fitting the scaler on all data (leakage)."""
    cfg, df, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = _setup()
    pre = fit_preprocessor(X_train, cfg)
    numeric_cols = list_feature_types(X_train)["numeric"]
    scaler = pre.named_transformers_["numeric"].named_steps["scale"]

    all_data_mean = pd.concat([X_train, X_val, X_test])[numeric_cols]
    all_data_mean = all_data_mean.fillna(all_data_mean.median()).mean().values
    train_mean = X_train[numeric_cols].fillna(X_train[numeric_cols].median()).mean().values

    assert np.allclose(scaler.mean_, train_mean, rtol=1e-2)
    assert not np.allclose(scaler.mean_, all_data_mean, rtol=1e-6)
    print("PASS: scaler was fit on train only, not the full dataset")


def test_pitfall_no_train_serve_drift():
    """Pitfall: train/serve preprocessing drift."""
    cfg, df, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = _setup()
    pre = fit_preprocessor(X_train, cfg)
    path = cfg.preprocessor_dir / "test_drift_check.joblib"
    save_preprocessor(pre, path)
    reloaded = load_preprocessor(path)

    live_output = transform(pre, X_test)
    served_output = transform(reloaded, X_test)
    assert np.allclose(live_output, served_output)
    print("PASS: reloaded (served) preprocessor output identical to train-time output")


def test_pitfall_missing_values_imputed_not_dropped():
    """Pitfall: dropping rows with missing values instead of handling them."""
    cfg, df, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = _setup()
    assert X_train.isna().sum().sum() > 0, "test setup expected some missing values"
    pre = fit_preprocessor(X_train, cfg)
    out = transform(pre, X_train)
    assert out.shape[0] == len(X_train), "row count changed — rows were dropped"
    assert not np.isnan(out).any(), "NaNs remain — not actually imputed"
    print("PASS: missing values imputed in-pipeline, zero rows dropped")


def test_leakage_guard_raises_on_synthetic_leak():
    """Confirm verify_no_leakage actually catches a real leak, not just passes vacuously."""
    cfg, df, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = _setup()
    # deliberately fit on train+val combined to simulate the leak this guard should catch
    leaky_pre = fit_preprocessor(pd.concat([X_train, X_val]), cfg)
    try:
        verify_no_leakage(leaky_pre, X_train, X_val)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "verify_no_leakage failed to catch a preprocessor fit on train+val"
    print("PASS: leakage guard correctly rejects a preprocessor fit on train+val")


def test_edge_case_missing_preprocessor_file():
    try:
        load_preprocessor(Path("/tmp/does_not_exist_preprocessor.joblib"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("PASS: loading a missing preprocessor artifact raises clearly")


def test_edge_case_empty_dataframe_transform():
    cfg, df, ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = _setup()
    pre = fit_preprocessor(X_train, cfg)
    try:
        transform(pre, X_train.iloc[0:0])
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: transforming an empty dataframe raises clearly")


if __name__ == "__main__":
    test_pitfall_scaler_not_fit_on_all_data()
    test_pitfall_no_train_serve_drift()
    test_pitfall_missing_values_imputed_not_dropped()
    test_leakage_guard_raises_on_synthetic_leak()
    test_edge_case_missing_preprocessor_file()
    test_edge_case_empty_dataframe_transform()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
