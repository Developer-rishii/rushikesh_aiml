"""
Tests for Task 12. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_calibrated_classifier.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.build import build_base_pipeline
from src.models.calibrate import fit_calibrated_variants, evaluate_calibration_quality
from src.evaluation.stability import evaluate_segments
from src.serving.package import load_serving_package

_CACHE = {}


def _shared_fit():
    if "variants" not in _CACHE:
        cfg = load_config()
        raw = load_dataframe(cfg)
        splits = split_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
        base = build_base_pipeline(cfg)
        base.fit(X_train, y_train)
        variants = fit_calibrated_variants(base, X_train, y_train, cfg)
        _CACHE.update(cfg=cfg, splits=splits, base=base, variants=variants)
    return _CACHE["cfg"], _CACHE["splits"], _CACHE["base"], _CACHE["variants"]


def test_live_end_to_end_run():
    from src.run_calibrated_classifier import main
    result = main()
    assert "chosen_threshold" in result
    assert Path(result["serving_package_paths"]["model_path"]).exists()
    assert result["serving_reload_verified"] is True
    print(f"PASS: live end-to-end run — calibration={result['selected_calibration_method']}, "
          f"threshold={result['chosen_threshold']}, operating-point missed-malignancy-rate="
          f"{result['operating_point']['expected_missed_malignancy_rate']}")


def test_pitfall_calibration_quality_measured_not_assumed():
    """Pitfall: Uncalibrated probabilities used as if exact."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base, variants = _shared_fit()
    brier = evaluate_calibration_quality(variants, base, X_val, y_val)
    assert "uncalibrated" in brier and "sigmoid" in brier and "isotonic" in brier
    assert all(isinstance(v, float) for v in brier.values())
    print(f"PASS: Brier score actually computed for uncalibrated + both calibration methods: {brier} "
          f"— the choice of which to use is evidence-based, not assumed")


def test_pitfall_segment_failure_would_be_caught():
    """Pitfall: Hidden per-segment failure."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base, variants = _shared_fit()
    proba = base.predict_proba(X_val)[:, 1]
    result = evaluate_segments(X_val, y_val, proba, 0.5, cfg)
    assert set(result["segments"].keys()) == set(cfg.segment_labels)
    assert "flagged_low_recall_segments" in result
    cfg.min_segment_recall = 1.01  # impossible bar, to prove the flag actually fires
    forced = evaluate_segments(X_val, y_val, proba, 0.5, cfg)
    assert len(forced["flagged_low_recall_segments"]) > 0, (
        "flagging mechanism never fires even with an impossible threshold — it's not wired up"
    )
    print(f"PASS: per-segment recall computed for all {len(result['segments'])} segments "
          f"AND the flagging mechanism verified to actually fire when a segment underperforms")


def test_pitfall_operating_point_is_documented():
    """Pitfall: No documented operating point."""
    from src.run_calibrated_classifier import main
    result = main()
    op = result["operating_point"]
    required_keys = {"calibration_method", "threshold", "test_metrics", "test_confusion_matrix",
                      "expected_missed_malignancy_rate", "expected_unnecessary_biopsy_rate"}
    assert required_keys <= op.keys()
    print(f"PASS: operating point fully documented with threshold={op['threshold']} and "
          f"expected error rates (missed_malignancy={op['expected_missed_malignancy_rate']}, "
          f"unnecessary_biopsy={op['expected_unnecessary_biopsy_rate']})")


def test_edge_case_unknown_calibration_method_raises():
    cfg = load_config()
    cfg.calibration_methods = ["not_a_real_method"]
    raw = load_dataframe(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(raw, cfg)
    base = build_base_pipeline(cfg)
    base.fit(X_train, y_train)
    try:
        fit_calibrated_variants(base, X_train, y_train, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown calibration method raises a clear error")


def test_edge_case_missing_serving_package_raises():
    try:
        load_serving_package(Path("/tmp/does_not_exist_serving_pkg"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("PASS: loading a missing/incomplete serving package raises clearly")


def test_edge_case_segment_feature_missing_raises():
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base, variants = _shared_fit()
    cfg.segment_feature = "not_a_real_column"
    proba = base.predict_proba(X_val)[:, 1]
    try:
        evaluate_segments(X_val, y_val, proba, 0.5, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: a missing segment feature raises clearly instead of silently skipping the fairness check")


if __name__ == "__main__":
    test_pitfall_calibration_quality_measured_not_assumed()
    test_pitfall_segment_failure_would_be_caught()
    test_edge_case_unknown_calibration_method_raises()
    test_edge_case_missing_serving_package_raises()
    test_edge_case_segment_feature_missing_raises()
    test_live_end_to_end_run()
    test_pitfall_operating_point_is_documented()
    print("\nALL TESTS PASSED")
