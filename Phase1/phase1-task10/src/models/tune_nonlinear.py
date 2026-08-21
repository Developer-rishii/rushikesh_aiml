"""
models/tune_nonlinear.py — Step 4: regularise/tune the non-linear model
to control overfitting, via the same train-only CV-search discipline as
Task 9 (never touches val/test).
"""
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.models.build import build_nonlinear_pipeline

log = logging.getLogger("src.models.tune_nonlinear")


def tune_nonlinear_model(X_train: pd.DataFrame, y_train, cfg):
    base_pipeline = build_nonlinear_pipeline(cfg)
    cv = StratifiedKFold(n_splits=cfg.nonlinear_cv_folds, shuffle=True, random_state=cfg.seed)
    search = GridSearchCV(
        base_pipeline, param_grid=cfg.nonlinear_search_space, scoring=cfg.nonlinear_scoring,
        cv=cv, n_jobs=-1, refit=True, return_train_score=True,
    )
    log.info("[Step 4] Regularisation search over %s combinations, cv_folds=%s, scoring=%s",
              _grid_size(cfg.nonlinear_search_space), cfg.nonlinear_cv_folds, cfg.nonlinear_scoring)
    search.fit(X_train, y_train)  # train-only, same guard as Task 9

    best_row_idx = search.best_index_
    cv_results = pd.DataFrame(search.cv_results_)
    train_val_gap = float(
        cv_results.loc[best_row_idx, "mean_train_score"] - cv_results.loc[best_row_idx, "mean_test_score"]
    )
    log.info("[Step 4] Best regularised config: %s (CV %s=%.4f, train-vs-CV gap=%.4f)",
              search.best_params_, cfg.nonlinear_scoring, search.best_score_, train_val_gap)
    return search, train_val_gap


def _grid_size(grid: dict) -> int:
    size = 1
    for v in grid.values():
        size *= len(v)
    return size
