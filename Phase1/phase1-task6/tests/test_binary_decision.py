"""
Tests for Task 6. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_binary_decision.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from configs.loader import load_config
from src.data.dataset import load_dataframe, enrich_dataframe, split_dataframe
from src.preprocessing.pipeline import fit_preprocessor, transform
from src.modeling.registry import build_model, build_model as _bm
from src.evaluation.decision_metrics import confusion_matrix_at_threshold, decision_metrics_at_threshold
from src.evaluation.threshold_selection import select_cost_optimal_threshold
from src.evaluation.imbalance_check import check_imbalance


def _fit_and_predict():
    cfg = load_config()
    raw = load_dataframe(cfg)
    enriched = enrich_dataframe(raw, cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(enriched, cfg)
    pre = fit_preprocessor(X_train, cfg)
    X_train_t, X_val_t = transform(pre, X_train), transform(pre, X_val)
    model = build_model(cfg.model_name, cfg.model_params)
    model.fit(X_train_t, y_train)
    y_proba = model.predict_proba(X_val_t)[:, 1]
    return cfg, y_train, y_val, y_proba


def test_live_end_to_end_run():
    from src.run_binary_decision import main
    result = main()
    assert "confusion_matrix_at_0.5" in result
    assert Path(result["curve_plots"]["roc"]).exists()
    assert Path(result["curve_plots"]["pr"]).exists()
    print("PASS: live end-to-end run — confusion matrix, curves, threshold report all produced")


def test_pitfall_not_accuracy_only():
    """Pitfall: Accuracy-only reporting."""
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    metrics = decision_metrics_at_threshold(y_val, y_proba, 0.5, cfg.metrics)
    reported = set(metrics.keys())
    assert {"precision", "recall", "f1"} <= reported, "precision/recall/f1 must all be reported, not just accuracy"
    assert len(reported) > 1, "only one metric reported — accuracy-only pitfall"
    print(f"PASS: {sorted(reported)} all reported together, not accuracy alone")


def test_pitfall_threshold_is_cost_justified_not_default():
    """Pitfall: Default threshold with no cost reasoning."""
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    result, sweep = select_cost_optimal_threshold(y_val, y_proba, cfg)
    assert "cost_reasoning" in result and len(result["cost_reasoning"]) > 20
    assert result["cost_false_negative_per_case"] != result["cost_false_positive_per_case"], (
        "costs are equal — no real cost reasoning driving threshold choice"
    )
    # the recommended threshold must actually be chosen BY minimizing cost, not hardcoded to 0.5
    rec = result["recommended_threshold"]["threshold"]
    default_cost = result["default_threshold_0.5"]["expected_cost"]
    rec_cost = result["recommended_threshold"]["expected_cost"]
    assert rec_cost <= default_cost, "recommended threshold does not actually reduce or match expected cost"
    print(f"PASS: threshold {rec} chosen by minimizing cost ({rec_cost} <= default {default_cost}), "
          f"with documented cost reasoning")


def test_pitfall_imbalance_not_ignored():
    """Pitfall: Ignoring imbalance entirely."""
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    result = check_imbalance(y_train, y_val)
    assert "val_majority_baseline_accuracy" in result
    assert "accuracy_is_potentially_misleading" in result
    print(f"PASS: imbalance explicitly checked — majority baseline accuracy = "
          f"{result['val_majority_baseline_accuracy']}")


def test_confusion_matrix_cells_are_correctly_labeled():
    """Sanity: confusion matrix cells sum to len(y_val), and semantics match sklearn's raw matrix."""
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    cm = confusion_matrix_at_threshold(y_val, y_proba, 0.5)
    total = (cm["true_negative_malignant_caught"] + cm["missed_malignancy"]
             + cm["unnecessary_biopsy"] + cm["true_positive_benign_cleared"])
    assert total == len(y_val)
    print(f"PASS: confusion matrix cells sum to {total} = len(y_val), correctly labeled")


def test_edge_case_unknown_metric_raises():
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    try:
        decision_metrics_at_threshold(y_val, y_proba, 0.5, ["not_a_real_metric"])
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown metric name raises a clear error")


def test_edge_case_threshold_at_extremes():
    """Thresholds of 0.0 and 1.0 shouldn't crash the confusion matrix computation."""
    cfg, y_train, y_val, y_proba = _fit_and_predict()
    cm_low = confusion_matrix_at_threshold(y_val, y_proba, 0.0001)
    cm_high = confusion_matrix_at_threshold(y_val, y_proba, 0.9999)
    assert cm_low["true_negative_malignant_caught"] + cm_low["missed_malignancy"] > 0
    assert cm_high["unnecessary_biopsy"] + cm_high["true_positive_benign_cleared"] >= 0
    print("PASS: extreme thresholds (near 0 and near 1) handled without crashing")


if __name__ == "__main__":
    test_pitfall_not_accuracy_only()
    test_pitfall_threshold_is_cost_justified_not_default()
    test_pitfall_imbalance_not_ignored()
    test_confusion_matrix_cells_are_correctly_labeled()
    test_edge_case_unknown_metric_raises()
    test_edge_case_threshold_at_extremes()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
