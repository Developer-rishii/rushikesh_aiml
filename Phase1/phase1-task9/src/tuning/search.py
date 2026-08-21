"""
tuning/search.py — Steps 2-4 of the build pipeline:
  2. Choose a search strategy and CV scheme.
  3. Run the search, scoring by the business metric.
  4. Select settings by validated (CV) score, never training score.

CRITICAL: this module is only ever handed X_train/y_train. It has no
parameter, return path, or import that could accept test data — the
structural guard against the #1 pitfall ("Tuning on the test set").
"""
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.tuning.model import build_pipeline

log = logging.getLogger("src.tuning.search")


def run_search(X_train: pd.DataFrame, y_train, cfg):
    """Runs entirely on TRAINING data via internal k-fold CV — never
    sees X_val or X_test. Returns the fitted search object."""
    base_pipeline = build_pipeline(cfg, cfg.default_params)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    if cfg.search_strategy == "grid":
        search = GridSearchCV(
            base_pipeline, param_grid=cfg.param_grid, scoring=cfg.scoring,
            cv=cv, n_jobs=-1, refit=True, return_train_score=True,
        )
    elif cfg.search_strategy == "random":
        search = RandomizedSearchCV(
            base_pipeline, param_distributions=cfg.param_grid, n_iter=cfg.n_random_iter,
            scoring=cfg.scoring, cv=cv, n_jobs=-1, refit=True,
            random_state=cfg.seed, return_train_score=True,
        )
    else:
        raise ValueError(f"Unknown search strategy '{cfg.search_strategy}'. Use 'grid' or 'random'.")

    log.info("[Step 2] Search strategy=%s, cv_folds=%s (StratifiedKFold), scoring=%s, "
              "grid size=%s combinations", cfg.search_strategy, cfg.cv_folds, cfg.scoring,
              _grid_size(cfg.param_grid) if cfg.search_strategy == "grid" else cfg.n_random_iter)

    search.fit(X_train, y_train)  # <-- ONLY X_train/y_train ever touch this call

    log.info("[Step 3/4] Search complete. Best CV %s=%.4f (selected by validated CV score, "
              "not training score) with params=%s",
              cfg.scoring, search.best_score_, search.best_params_)
    return search


def _grid_size(param_grid: dict) -> int:
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size


def summarize_cv_results(search) -> pd.DataFrame:
    """Full CV leaderboard, sorted by mean validated score — the "CV
    results" deliverable, not just the single best number."""
    results = pd.DataFrame(search.cv_results_)
    cols = [c for c in results.columns if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "mean_train_score", "rank_test_score")]
    return results[cols].sort_values("rank_test_score").reset_index(drop=True)
