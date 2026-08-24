"""
validation/nested_cv.py — Step 4: nested CV for the model whose
hyperparameters are tuned, so tuning never touches the same fold split
used to report its generalisation score. This is the direct, structural
guard against the pitfall "Tuning and evaluating on the same folds":

  OUTER loop (cfg.outer_folds): held-out test fold, NEVER seen by the
  inner GridSearchCV for that iteration.
  INNER loop (cfg.inner_folds), inside GridSearchCV: only ever touches
  the outer loop's TRAINING portion — picks the best C using folds the
  outer test fold has no part in.

The reported nested-CV score is the outer-fold score on data the
hyperparameter search never saw, which is what makes it an honest
generalisation estimate rather than an optimistic, leaked one.
"""
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

from src.validation.models import build_pipeline

log = logging.getLogger("src.validation.nested_cv")


def run_nested_cv(X, y, cfg) -> dict:
    model_name = cfg.model_to_tune
    if model_name not in cfg.candidate_models:
        raise ValueError(f"nested_cv.model_to_tune='{model_name}' is not in candidate_models.")

    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.seed)
    inner_cv = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed)

    base_pipeline = build_pipeline(cfg, model_name)
    search = GridSearchCV(
        base_pipeline, param_grid=cfg.nested_param_grid, scoring=cfg.scoring,
        cv=inner_cv, n_jobs=-1, refit=True,
    )

    # cross_val_score with an unfit GridSearchCV as the "estimator" IS
    # nested CV: for each outer fold, .fit() re-runs the full inner grid
    # search on that fold's training portion only, then scores on the
    # outer fold's held-out portion -- the inner search literally cannot
    # see the outer test fold at any point.
    nested_scores = cross_val_score(search, X, y, cv=outer_cv, scoring=cfg.scoring, n_jobs=-1)

    # also report what the naive (non-nested) approach would have shown,
    # for direct comparison: tune once on the whole dataset, look at
    # that single best inner CV score -- typically optimistic.
    search.fit(X, y)
    naive_non_nested_score = float(search.best_score_)

    result = {
        "model_tuned": model_name,
        "param_grid": cfg.nested_param_grid,
        "outer_folds": cfg.outer_folds,
        "inner_folds": cfg.inner_folds,
        "nested_cv_fold_scores": [round(float(s), 4) for s in nested_scores],
        "nested_cv_mean": round(float(nested_scores.mean()), 4),
        "nested_cv_std": round(float(nested_scores.std()), 4),
        "naive_non_nested_best_score": round(naive_non_nested_score, 4),
        "optimism_gap": round(naive_non_nested_score - float(nested_scores.mean()), 4),
        "best_params_on_full_data": search.best_params_,
    }
    log.info("[Step 4] Nested CV: mean=%.4f std=%.4f | naive non-nested score=%.4f | optimism gap=%+.4f",
              result["nested_cv_mean"], result["nested_cv_std"],
              result["naive_non_nested_best_score"], result["optimism_gap"])
    return result
