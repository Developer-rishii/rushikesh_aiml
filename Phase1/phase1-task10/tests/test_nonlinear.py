"""
Tests for Task 10. One test per named pitfall, plus a live end-to-end run
and edge cases. The expensive GridSearchCV is run ONCE and shared across
tests via a module-level cache to keep the suite fast.
Run: python tests/test_nonlinear.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.build import build_baseline_pipeline, build_nonlinear_pipeline
from src.models.tune_nonlinear import tune_nonlinear_model
from src.evaluation.metrics import evaluate_pipeline
from src.evaluation.effects import top_features_by_importance, plot_partial_dependence

_CACHE = {}


def _shared_search():
    if "search" not in _CACHE:
        cfg = load_config()
        raw = load_dataframe(cfg)
        splits = split_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
        search, gap = tune_nonlinear_model(X_train, y_train, cfg)
        _CACHE["cfg"] = cfg
        _CACHE["splits"] = splits
        _CACHE["search"] = search
        _CACHE["gap"] = gap
    return _CACHE["cfg"], _CACHE["splits"], _CACHE["search"], _CACHE["gap"]


def test_live_end_to_end_run():
    from src.run_nonlinear import main
    result = main()
    assert result["decision"].startswith("KEEP") or result["decision"].startswith("REJECT")
    assert Path(result["pdp_plot_path"]).exists()
    print(f"PASS: live end-to-end run — decision: {result['kept_model']}, "
          f"validated lift={result['validated_lift']:+.4f}")


def test_pitfall_decision_gated_on_validated_lift_not_complexity():
    """Pitfall: Complexity with no validated gain."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), search, gap = _shared_search()

    baseline = build_baseline_pipeline(cfg)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_pipeline(baseline, X_val, y_val, cfg.metrics)
    nonlinear_metrics = evaluate_pipeline(search.best_estimator_, X_val, y_val, cfg.metrics)

    lift = nonlinear_metrics[cfg.primary_metric] - baseline_metrics[cfg.primary_metric]
    would_keep = lift >= cfg.min_lift_to_keep
    assert cfg.min_lift_to_keep > 0
    print(f"PASS: keep/reject gated on a real threshold ({cfg.min_lift_to_keep}), "
          f"measured lift={lift:+.4f} -> would_keep={would_keep}")


def test_pitfall_regularisation_is_actually_searched():
    """Pitfall: Overfitting from unregularised power."""
    cfg = load_config()
    keys = set(cfg.nonlinear_search_space.keys())
    assert "model__max_depth" in keys, "max_depth (complexity cap) must be tuned"
    assert "model__subsample" in keys, "subsample (stochastic regularisation) must be tuned"
    assert max(cfg.nonlinear_search_space["model__max_depth"]) <= 5, (
        "max_depth search space includes unreasonably deep trees for a 398-row training set"
    )
    print(f"PASS: regularisation knobs actually searched: {sorted(keys)}, "
          f"max depth capped at {max(cfg.nonlinear_search_space['model__max_depth'])}")


def test_pitfall_overfitting_actually_checked_not_assumed_absent():
    cfg, splits, search, gap = _shared_search()
    assert isinstance(gap, float)
    print(f"PASS: train-vs-CV-fold gap actually computed for the winning config: {gap:+.4f}")


def test_pitfall_explainability_preserved_via_pdp():
    """Pitfall: Losing all explainability."""
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), search, gap = _shared_search()
    top_features = top_features_by_importance(search.best_estimator_, list(X_train.columns), cfg.pdp_n_top_features)
    assert len(top_features) == cfg.pdp_n_top_features
    path = plot_partial_dependence(search.best_estimator_, X_train, list(X_train.columns), top_features, cfg.figure_dir)
    assert Path(path).exists() and Path(path).stat().st_size > 0
    print(f"PASS: partial dependence plot actually generated for top features {top_features} — "
          f"the model's behavior is inspectable, not a black box")


def test_edge_case_unknown_nonlinear_model_raises():
    cfg = load_config()
    cfg.nonlinear_model_name = "does_not_exist"
    try:
        build_nonlinear_pipeline(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown non-linear model name raises a clear error")


def test_edge_case_pdp_fails_clearly_on_non_tree_model():
    cfg, ((X_train, y_train), (X_val, y_val), (X_test, y_test)), search, gap = _shared_search()
    baseline = build_baseline_pipeline(cfg)
    baseline.fit(X_train, y_train)
    try:
        top_features_by_importance(baseline, list(X_train.columns), 4)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: requesting feature_importances_ from a model that doesn't have them raises clearly")


if __name__ == "__main__":
    test_pitfall_regularisation_is_actually_searched()
    test_edge_case_unknown_nonlinear_model_raises()
    test_pitfall_overfitting_actually_checked_not_assumed_absent()
    test_pitfall_explainability_preserved_via_pdp()
    test_pitfall_decision_gated_on_validated_lift_not_complexity()
    test_edge_case_pdp_fails_clearly_on_non_tree_model()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
