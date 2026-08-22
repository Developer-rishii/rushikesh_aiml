"""
Tests for Task 11. One test per named pitfall, plus a live end-to-end run
and edge cases. The expensive base-model fits are cached and shared.
Run: python tests/test_ensemble.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.base import build_all_base_pipelines, build_base_pipeline
from src.models.ensemble import build_stacking_ensemble
from src.evaluation.diversity import compute_error_overlap, diversity_verdict
from src.evaluation.latency import measure_inference_latency

_CACHE = {}


def _shared_fit():
    if "base_pipelines" not in _CACHE:
        cfg = load_config()
        raw = load_dataframe(cfg)
        splits = split_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
        base_pipelines = build_all_base_pipelines(cfg)
        base_preds = {}
        for name, pipe in base_pipelines.items():
            pipe.fit(X_train, y_train)
            base_preds[name] = pipe.predict(X_val)
        _CACHE.update(cfg=cfg, splits=splits, base_pipelines=base_pipelines, base_preds=base_preds)
    return _CACHE["cfg"], _CACHE["splits"], _CACHE["base_pipelines"], _CACHE["base_preds"]


def test_live_end_to_end_run():
    from src.run_ensemble import main
    result = main()
    assert result["decision"].startswith("PREFER")
    assert "diversity_check" in result and "latency" in result
    print(f"PASS: live end-to-end run — kept model: {result['kept_model']}, "
          f"validated lift={result['validated_lift']:+.4f}")


def test_pitfall_diversity_actually_checked_not_assumed():
    """Pitfall: Ensembling near-identical models."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base_pipelines, base_preds = _shared_fit()
    overlap = compute_error_overlap(base_preds, y_val)
    verdict = diversity_verdict(overlap)
    assert len(overlap) == 3, "expected 3 pairwise comparisons among 3 base models"
    assert all("error_overlap_fraction" in v for v in overlap.values())
    print(f"PASS: pairwise error-overlap actually computed for all {len(overlap)} base-model pairs "
          f"(diversity_confirmed={verdict['diversity_confirmed']}), not assumed")


def test_pitfall_stacking_does_not_leak_across_folds():
    """Pitfall: Stacking that leaks across folds."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base_pipelines, base_preds = _shared_fit()
    stacking = build_stacking_ensemble(cfg)
    assert stacking.cv is not None and stacking.cv >= 2, (
        f"StackingClassifier.cv={stacking.cv} — a None/1 value would let base "
        f"models see their own training predictions as meta-features (leakage)"
    )
    stacking.fit(X_train, y_train)
    proba = stacking.predict_proba(X_val)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    print(f"PASS: StackingClassifier configured with internal cv={stacking.cv} "
          f"(guards against meta-model seeing leaked in-sample base predictions)")


def test_pitfall_latency_measured_not_ignored():
    """Pitfall: Ignoring inference cost."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base_pipelines, base_preds = _shared_fit()
    single = list(base_pipelines.values())[0]
    latency = measure_inference_latency(single, X_val, n_repeats=10)
    assert latency["mean_batch_latency_ms"] > 0
    assert latency["n_repeats"] == 10
    print(f"PASS: inference latency actually measured ({latency['mean_batch_latency_ms']}ms/batch), "
          f"not assumed negligible")


def test_edge_case_unknown_base_model_raises():
    cfg = load_config()
    try:
        build_base_pipeline(cfg, "does_not_exist")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown base model name raises a clear error")


def test_edge_case_unknown_meta_model_raises():
    cfg = load_config()
    cfg.stacking_meta_model = "does_not_exist"
    try:
        build_stacking_ensemble(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown stacking meta-model raises a clear error")


def test_edge_case_latency_on_empty_sample_raises():
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), base_pipelines, base_preds = _shared_fit()
    single = list(base_pipelines.values())[0]
    try:
        measure_inference_latency(single, X_val.iloc[0:0])
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: measuring latency on an empty sample raises clearly")


if __name__ == "__main__":
    test_pitfall_diversity_actually_checked_not_assumed()
    test_pitfall_stacking_does_not_leak_across_folds()
    test_pitfall_latency_measured_not_ignored()
    test_edge_case_unknown_base_model_raises()
    test_edge_case_unknown_meta_model_raises()
    test_edge_case_latency_on_empty_sample_raises()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
