"""
run_validation.py — Task 16's full flow, in the study guide's exact
step order:
  1. Choose an appropriate CV scheme (stratified/time-aware).
  2. Run K-Fold and collect per-fold scores.
  3. Report mean and spread, not just the best fold.
  4. Use nested CV if you also tuned hyperparameters.
  5. Compare candidate models on the same folds.
  6. Conclude which model generalises best.

Run: python -m src.run_validation
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_features_and_target
from src.validation.kfold_compare import compare_candidates_cv
from src.validation.nested_cv import run_nested_cv
from src.validation.select import select_most_generalising
from src.validation.plots import plot_fold_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_validation")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("[Step 1] Loaded config: %s. CV scheme: StratifiedKFold(%s) — stratified because "
              "the target is imbalanced (Task 2's ~63/37 split).", cfg, cfg.outer_folds)

    try:
        X, y = load_features_and_target(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    try:
        cv_comparison = compare_candidates_cv(X, y, cfg)
    except Exception as e:
        log.error("CV comparison stage failed: %s", e)
        sys.exit(1)
    plot_path = plot_fold_scores(cv_comparison["per_model_results"], cfg.figure_dir)

    try:
        nested_result = run_nested_cv(X, y, cfg)
    except ValueError as e:
        log.error("Nested CV stage failed: %s", e)
        sys.exit(1)

    selection = select_most_generalising(cv_comparison["per_model_results"], cfg)

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "seed": cfg.seed,
        "n_rows": len(X),
        "cv_comparison": cv_comparison,
        "nested_cv_for_tuned_model": nested_result,
        "selection": selection,
        "fold_scores_plot": plot_path,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "validation_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_validation.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Selected: %s. Report -> %s",
              result["runtime_seconds"], selection["selected_model"], cfg.report_dir / "validation_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
