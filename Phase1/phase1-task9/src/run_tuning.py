"""
run_tuning.py — Task 9's full flow, in the study guide's exact step order:
  1. Pick the hyperparameters that matter most for this model.
  2. Choose a search strategy and a CV scheme.
  3. Run the search, scoring by the business metric.
  4. Select settings by validated (not training) score.
  5. Confirm the gain holds on the held-out test set.
  6. Record the best config and the improvement.

Data flow discipline (the whole point of this task):
  X_train, y_train -> search.py (CV only, Steps 2-4)
  X_test, y_test   -> confirm.py (Step 5, touched exactly once, at the end)
  X_val is not used in this task at all — Task 8 already validated the
  default config on val; this task's job is CV-vs-test, not val.

Run: python -m src.run_tuning
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.tuning.search import run_search, summarize_cv_results
from src.tuning.confirm import confirm_test_gain

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_tuning")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- data + split ----
    try:
        df = load_dataframe(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(df, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 1] Search space (only hyperparameters affecting bias/variance trade-off): %s",
              cfg.param_grid)

    # ---- Steps 2-4: search (train-only, internal CV) ----
    try:
        search = run_search(X_train, y_train, cfg)
        cv_results = summarize_cv_results(search)
    except ValueError as e:
        log.error("Search stage failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Search raised an unexpected error: %s", e)
        sys.exit(1)

    # ---- Step 5: confirm on held-out test set (touched exactly once) ----
    try:
        confirm_result, tuned_pipeline, default_pipeline = confirm_test_gain(
            X_train, y_train, X_test, y_test, search.best_params_, cfg
        )
    except Exception as e:
        log.error("Test-set confirmation failed: %s", e)
        sys.exit(1)

    # ---- overfitting-the-CV check (brainstorming Q1: tuning to validation noise?) ----
    best_row = cv_results.iloc[0]
    cv_test_gap = float(best_row["mean_train_score"] - best_row["mean_test_score"])
    overfitting_flag = cv_test_gap > 0.05
    log.info("CV train-vs-validation-fold gap for the winning config: %.4f (%s)",
              cv_test_gap, "POSSIBLE OVERFIT TO CV" if overfitting_flag else "healthy")

    # ---- Step 6: record best config + improvement ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(tuned_pipeline, cfg.artifact_dir / "tuned_pipeline.joblib")
    joblib.dump(default_pipeline, cfg.artifact_dir / "default_pipeline.joblib")
    cv_results.to_csv(cfg.report_dir / "cv_results.csv", index=False)

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "search_strategy": cfg.search_strategy,
        "cv_folds": cfg.cv_folds,
        "scoring": cfg.scoring,
        "param_grid": cfg.param_grid,
        "best_cv_score": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "cv_train_val_gap_for_winner": round(cv_test_gap, 4),
        "possible_cv_overfit": overfitting_flag,
        "test_set_confirmation": confirm_result,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "tuning_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_tuning.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("[Step 6] Done in %ss. Best config + confirmed test gain -> %s",
              result["runtime_seconds"], cfg.report_dir / "tuning_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
