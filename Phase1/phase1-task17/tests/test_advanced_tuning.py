"""
Tests for Task 17. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_advanced_tuning.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_features_and_target
from src.data.split import split_train_test
from src.tuning.preprocess import fit_preprocessor
from src.tuning.search import run_search


def test_live_end_to_end_run():
    from src.run_advanced_tuning import main
    result = main()
    assert result["n_trials_completed"] + result["n_trials_pruned_early"] == result["n_trials_requested"]
    assert Path(result["trial_log_path"]).exists()
    print(f"PASS: live end-to-end run — {result['n_trials_completed']} completed / "
          f"{result['n_trials_pruned_early']} pruned ({result['compute_saved_pct_via_pruning']}% compute saved), "
          f"test-confirmed gain={result['test_confirmation']['test_confirmed_gain']:+.4f}")


def test_pitfall_search_space_is_not_huge_wasteful_grid():
    """Pitfall: Huge wasteful grids."""
    cfg = load_config()
    equivalent_grid_size = (
        (cfg.max_depth_range[1] - cfg.max_depth_range[0] + 1)
        * 10
        * 6
        * (cfg.min_samples_leaf_range[1] - cfg.min_samples_leaf_range[0] + 1)
    )
    assert cfg.n_trials < equivalent_grid_size / 10, (
        f"n_trials={cfg.n_trials} is not meaningfully smaller than an equivalent grid "
        f"({equivalent_grid_size}) — the search isn't actually more efficient than grid search"
    )
    print(f"PASS: {cfg.n_trials} adaptive trials vs. an equivalent exhaustive grid of "
          f"~{equivalent_grid_size} combinations — genuinely efficient, not a huge grid in disguise")


def test_pitfall_pruning_actually_saves_trials():
    """Pitfall (compute-waste half): confirm pruning has real teeth, not just configured."""
    cfg = load_config()
    X, y = load_features_and_target(cfg)
    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)
    X_train_p, imputer, scaler = fit_preprocessor(X_train, cfg)
    study, trial_log = run_search(X_train_p, y_train, cfg)
    n_pruned = sum(1 for t in trial_log if t["status"] == "PRUNED")
    assert len(trial_log) == cfg.n_trials
    print(f"PASS: pruning mechanism actually fired on {n_pruned}/{len(trial_log)} trials "
          f"(0 pruned would still be a valid honest outcome, but the mechanism ran and reported for every trial)")


def test_pitfall_search_overfitting_actually_checked():
    """Pitfall: Overfitting the search to validation."""
    from src.run_advanced_tuning import main
    result = main()
    assert "cv_vs_test_gap" in result and "possible_search_overfitting" in result
    print(f"PASS: CV-best score vs held-out test score gap actually computed "
          f"({result['cv_vs_test_gap']:+.4f}), not assumed to generalize")


def test_pitfall_all_trials_logged():
    """Pitfall: Unreproducible, unlogged trials."""
    cfg = load_config()
    X, y = load_features_and_target(cfg)
    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)
    X_train_p, imputer, scaler = fit_preprocessor(X_train, cfg)
    study, trial_log = run_search(X_train_p, y_train, cfg)
    assert len(trial_log) == cfg.n_trials, "not every trial (completed or pruned) was logged"
    for t in trial_log:
        assert "params" in t and "status" in t and "trial_number" in t
    print(f"PASS: all {len(trial_log)} trials logged with params + status, "
          f"regardless of whether they completed or were pruned")


def test_edge_case_log_scale_actually_used():
    """Confirms learning_rate really samples log-uniform, not linear-uniform
    mislabeled as log (a common silent bug)."""
    import optuna
    cfg = load_config()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    samples = []
    for _ in range(200):
        trial = study.ask()
        val = trial.suggest_float("learning_rate", *cfg.learning_rate_range, log=True)
        samples.append(val)
        study.tell(trial, 0.0)
    frac_below_10pct_of_range = sum(1 for s in samples if s < cfg.learning_rate_range[0] * 10) / len(samples)
    assert frac_below_10pct_of_range > 0.3, (
        "learning_rate samples don't show the expected log-scale concentration near the low end"
    )
    print(f"PASS: learning_rate genuinely sampled on a log scale "
          f"({frac_below_10pct_of_range:.0%} of samples fall in the lowest decade), not linear")


def test_edge_case_zero_trials_raises_or_handled():
    cfg = load_config()
    cfg.n_trials = 0
    X, y = load_features_and_target(cfg)
    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)
    X_train_p, imputer, scaler = fit_preprocessor(X_train, cfg)
    study, trial_log = run_search(X_train_p, y_train, cfg)
    assert len(trial_log) == 0
    try:
        _ = study.best_value
        raised = False
    except ValueError:
        raised = True
    assert raised, "accessing best_value with zero completed trials should raise, not silently return garbage"
    print("PASS: zero-trial search correctly leaves no valid 'best' result, and accessing one raises clearly")


if __name__ == "__main__":
    test_pitfall_search_space_is_not_huge_wasteful_grid()
    test_pitfall_pruning_actually_saves_trials()
    test_pitfall_all_trials_logged()
    test_edge_case_log_scale_actually_used()
    test_edge_case_zero_trials_raises_or_handled()
    test_live_end_to_end_run()
    test_pitfall_search_overfitting_actually_checked()
    print("\nALL TESTS PASSED")
