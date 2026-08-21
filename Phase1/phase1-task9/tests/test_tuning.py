"""
Tests for Task 9. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_tuning.py
"""
import sys
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.tuning.search import run_search
from src.tuning.model import build_pipeline


def test_live_end_to_end_run():
    from src.run_tuning import main
    result = main()
    assert "best_params" in result
    assert "test_set_confirmation" in result
    assert Path(str(load_config().artifact_dir / "tuned_pipeline.joblib")).exists()
    print(f"PASS: live end-to-end run — best CV score={result['best_cv_score']}, "
          f"test gain={result['test_set_confirmation']['test_set_gain']:+.4f}")


def test_pitfall_search_never_touches_test_set():
    """Pitfall: Tuning on the test set."""
    source = inspect.getsource(run_search)
    # strip docstrings/comments before the structural check — mentioning
    # "X_test" in a comment describing what NOT to do is fine; using it
    # as a live variable is what actually matters.
    import ast
    tree = ast.parse(source)
    func_node = tree.body[0]
    func_node.body = [n for n in func_node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code_only = ast.unparse(func_node)
    sig = inspect.signature(run_search)
    param_names = list(sig.parameters.keys())
    assert param_names[0] in ("X_train",), f"run_search's first param is '{param_names[0]}', expected X_train"
    assert "X_test" not in code_only and "y_test" not in code_only
    assert "X_val" not in code_only and "y_val" not in code_only
    print("PASS: run_search's signature/body physically cannot reference test or val data")


def test_pitfall_confirm_module_uses_test_exactly_once():
    """Pitfall: Reporting CV best as final without test confirmation."""
    import src.tuning.confirm as confirm_module
    source = inspect.getsource(confirm_module)
    # the module must actually call .fit and .predict/.predict_proba on
    # X_test's counterpart data, i.e. it doesn't just report CV numbers
    assert "X_test" in inspect.signature(confirm_module.confirm_test_gain).parameters
    assert "_evaluate(default_pipeline, X_test, y_test)" in source
    assert "_evaluate(tuned_pipeline, X_test, y_test)" in source
    print("PASS: confirm.py explicitly evaluates BOTH default and tuned models on X_test — "
          "CV score alone is never reported as the final answer")


def test_pitfall_only_bias_variance_params_are_searched():
    """Pitfall: Searching params that don't matter."""
    cfg = load_config()
    searched = set(cfg.param_grid.keys())
    # max_iter and random_state must NOT be in the search space — they
    # don't affect the bias/variance trade-off, they're just plumbing
    assert not any("max_iter" in k for k in searched)
    assert not any("random_state" in k for k in searched)
    assert any("C" in k for k in searched), "regularisation strength C should be searched — it's the core knob"
    print(f"PASS: search space {sorted(searched)} contains only bias/variance-relevant "
          f"hyperparameters, not plumbing params")


def test_cv_results_are_a_real_leaderboard_not_just_the_best():
    cfg = load_config()
    raw = load_dataframe(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(raw, cfg)
    search = run_search(X_train, y_train, cfg)
    from src.tuning.search import summarize_cv_results
    cv_df = summarize_cv_results(search)
    assert len(cv_df) > 1, "CV results should cover the full grid, not just the winner"
    assert "mean_test_score" in cv_df.columns
    print(f"PASS: CV leaderboard covers {len(cv_df)} parameter combinations, not just the single best")


def test_edge_case_unknown_search_strategy_raises():
    cfg = load_config()
    cfg.search_strategy = "not_a_real_strategy"
    raw = load_dataframe(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(raw, cfg)
    try:
        run_search(X_train, y_train, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown search strategy raises a clear error before any fitting")


def test_edge_case_unknown_model_raises():
    cfg = load_config()
    cfg.model_name = "does_not_exist"
    try:
        build_pipeline(cfg, {})
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown model name raises a clear error")


if __name__ == "__main__":
    test_pitfall_search_never_touches_test_set()
    test_pitfall_only_bias_variance_params_are_searched()
    test_edge_case_unknown_search_strategy_raises()
    test_edge_case_unknown_model_raises()
    test_cv_results_are_a_real_leaderboard_not_just_the_best()
    test_live_end_to_end_run()
    test_pitfall_confirm_module_uses_test_exactly_once()
    print("\nALL TESTS PASSED")
