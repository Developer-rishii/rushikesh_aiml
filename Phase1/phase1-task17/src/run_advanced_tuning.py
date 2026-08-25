"""
run_advanced_tuning.py — Task 17's full flow, in the study guide's exact
step order:
  1. Design a sensible search space with correct scales.
  2. Use Bayesian/efficient search with pruning.
  3. Score by robust CV on the business metric.
  4. Apply early stopping where supported.
  5. Confirm the winning config on the held-out test set.
  6. Log all trials for reproducibility.

Run: python -m src.run_advanced_tuning
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib

from configs.loader import load_config
from src.data.dataset import load_features_and_target
from src.data.split import split_train_test
from src.tuning.preprocess import fit_preprocessor, transform
from src.tuning.search import run_search
from src.tuning.confirm import confirm_on_test
from src.tuning.trial_log import save_trial_log, plot_optimization_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_advanced_tuning")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("[Step 1] Loaded config: %s. Search space: max_depth=%s (linear), "
              "learning_rate=%s (LOG scale), subsample=%s (linear), "
              "min_samples_leaf=%s (linear). n_estimators capped at %s but "
              "early-stopped per trial (Step 4).",
              cfg, cfg.max_depth_range, cfg.learning_rate_range, cfg.subsample_range,
              cfg.min_samples_leaf_range, cfg.n_estimators_max)

    try:
        X, y = load_features_and_target(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg)

    try:
        X_train_p, imputer, scaler = fit_preprocessor(X_train, cfg)
        X_test_p = transform(X_test, imputer, scaler)
    except ValueError as e:
        log.error("Preprocessing failed: %s", e)
        sys.exit(1)

    try:
        study, trial_log = run_search(X_train_p, y_train, cfg)
    except Exception as e:
        log.error("Search failed: %s", e)
        sys.exit(1)

    n_pruned = sum(1 for t in trial_log if t["status"] == "PRUNED")
    n_complete = len(trial_log) - n_pruned
    compute_saved_pct = round(100 * n_pruned / len(trial_log), 1) if trial_log else 0.0

    try:
        confirm_result, tuned_model, baseline_model = confirm_on_test(
            study.best_params, cfg, X_train_p, y_train, X_test_p, y_test
        )
    except Exception as e:
        log.error("Test-set confirmation failed: %s", e)
        sys.exit(1)

    cv_vs_test_gap = round(study.best_value - confirm_result["tuned_test_metrics"]["pr_auc"], 4)
    search_overfit_flag = cv_vs_test_gap > 0.05
    log.info("CV-best score vs test score for the winning config: gap=%+.4f (%s)",
              cv_vs_test_gap, "POSSIBLE SEARCH OVERFITTING" if search_overfit_flag else "healthy")

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    trial_log_path = save_trial_log(trial_log, cfg.report_dir)
    history_plot_path = plot_optimization_history(trial_log, cfg.figure_dir)

    joblib.dump(tuned_model, cfg.artifact_dir / "tuned_model.joblib")
    joblib.dump(baseline_model, cfg.artifact_dir / "baseline_model.joblib")
    joblib.dump({"imputer": imputer, "scaler": scaler}, cfg.artifact_dir / "preprocessor.joblib")

    param_importance = {}
    try:
        import optuna
        if n_complete >= 2:
            param_importance = optuna.importance.get_param_importances(study)
            param_importance = {k: round(float(v), 4) for k, v in param_importance.items()}
    except Exception as e:
        log.warning("Could not compute parameter importance: %s", e)

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "test": len(X_test)},
        "search_space": {
            "max_depth": cfg.max_depth_range, "learning_rate_log_scale": cfg.learning_rate_range,
            "subsample": cfg.subsample_range, "min_samples_leaf": cfg.min_samples_leaf_range,
            "n_estimators_ceiling": cfg.n_estimators_max,
        },
        "n_trials_requested": cfg.n_trials,
        "n_trials_completed": n_complete,
        "n_trials_pruned_early": n_pruned,
        "compute_saved_pct_via_pruning": compute_saved_pct,
        "best_cv_score": round(float(study.best_value), 4),
        "best_params": study.best_params,
        "param_importance": param_importance,
        "cv_vs_test_gap": cv_vs_test_gap,
        "possible_search_overfitting": search_overfit_flag,
        "test_confirmation": confirm_result,
        "trial_log_path": trial_log_path,
        "optimization_history_plot": history_plot_path,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "advanced_tuning_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_advanced_tuning.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. %s/%s trials completed (%s pruned, %s%% compute saved). "
              "Test-confirmed gain=%+.4f. Report -> %s",
              result["runtime_seconds"], n_complete, len(trial_log), n_pruned, compute_saved_pct,
              confirm_result["test_confirmed_gain"], cfg.report_dir / "advanced_tuning_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
