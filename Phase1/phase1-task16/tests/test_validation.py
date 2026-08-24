"""
Tests for Task 16. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_validation.py
"""
import sys
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_features_and_target
from src.validation.kfold_compare import compare_candidates_cv
from src.validation.select import select_most_generalising
from src.validation.models import build_pipeline

_CACHE = {}


def _shared_data():
    if "X" not in _CACHE:
        cfg = load_config()
        X, y = load_features_and_target(cfg)
        _CACHE.update(cfg=cfg, X=X, y=y)
    return _CACHE["cfg"], _CACHE["X"], _CACHE["y"]


def test_live_end_to_end_run():
    from src.run_validation import main
    result = main()
    assert result["selection"]["selected_model"] in result["cv_comparison"]["per_model_results"]
    assert Path(result["fold_scores_plot"]).exists()
    print(f"PASS: live end-to-end run — selected model: {result['selection']['selected_model']} "
          f"(mean={result['selection']['selected_mean']}, std={result['selection']['selected_std']})")


def test_pitfall_reports_spread_not_just_best_fold():
    """Pitfall: Reporting only the best fold."""
    cfg, X, y = _shared_data()
    comparison = compare_candidates_cv(X, y, cfg)
    for name, res in comparison["per_model_results"].items():
        assert len(res["fold_scores"]) == cfg.outer_folds, f"{name} does not report every fold's score"
        assert "std" in res and "min" in res and "max" in res, f"{name} report is missing spread statistics"
        assert res["max"] >= res["mean"] >= res["min"], "mean should sit within [min, max] of reported folds"
    print(f"PASS: every candidate reports all {cfg.outer_folds} fold scores plus mean/std/min/max — "
          f"the best fold alone is never the headline number")


def test_pitfall_folds_are_actually_stratified():
    """Pitfall: Non-stratified folds on imbalanced data."""
    cfg, X, y = _shared_data()
    comparison = compare_candidates_cv(X, y, cfg)
    check = comparison["stratification_check"]
    assert check["max_deviation_from_overall"] < 0.05, (
        f"per-fold positive rate deviates {check['max_deviation_from_overall']} from the overall rate — "
        f"folds are not properly stratified"
    )
    print(f"PASS: per-fold positive rate stays within {check['max_deviation_from_overall']} of the overall "
          f"{check['overall_positive_rate']} rate across all folds — genuinely stratified, not just labeled so")


def test_pitfall_nested_cv_structurally_cannot_leak():
    """Pitfall: Tuning and evaluating on the same folds."""
    from src.validation.nested_cv import run_nested_cv
    source = inspect.getsource(run_nested_cv)
    assert "cross_val_score(search, X, y, cv=outer_cv" in source, (
        "nested CV is not using an unfit GridSearchCV as the outer-loop estimator — "
        "this is the structural pattern that prevents leakage"
    )
    cfg, X, y = _shared_data()
    result = run_nested_cv(X, y, cfg)
    assert "optimism_gap" in result, "no comparison against the naive (leaky) approach was computed"
    print(f"PASS: nested CV structurally implemented via cross_val_score(GridSearchCV, ...) — "
          f"measured optimism gap vs naive tuning = {result['optimism_gap']:+.4f}")


def test_selection_prefers_low_variance_over_raw_mean_when_configured():
    cfg, X, y = _shared_data()
    fake_results = {
        "high_mean_high_var": {"mean": 0.99, "std": 0.15, "min": 0.7, "max": 1.0},
        "lower_mean_stable": {"mean": 0.95, "std": 0.01, "min": 0.93, "max": 0.97},
    }
    cfg.max_acceptable_std = 0.03
    selection = select_most_generalising(fake_results, cfg)
    assert selection["selected_model"] == "lower_mean_stable", (
        "selection picked the higher-mean-but-unstable model despite a variance cap — "
        "'most generalising' is not actually being enforced"
    )
    print(f"PASS: with a variance cap configured, selection correctly prefers the stable "
          f"lower-mean model over the unstable higher-mean one")


def test_edge_case_unknown_model_raises():
    cfg, X, y = _shared_data()
    try:
        build_pipeline(cfg, "does_not_exist")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown candidate model name raises a clear error")


if __name__ == "__main__":
    test_pitfall_reports_spread_not_just_best_fold()
    test_pitfall_folds_are_actually_stratified()
    test_pitfall_nested_cv_structurally_cannot_leak()
    test_selection_prefers_low_variance_over_raw_mean_when_configured()
    test_edge_case_unknown_model_raises()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
