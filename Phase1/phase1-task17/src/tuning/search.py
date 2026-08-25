"""
tuning/search.py — Steps 1-4: the Optuna objective and study runner.

  Step 1 (search space + scales): learning_rate sampled log-uniform
    (suggest_float(..., log=True)) since it spans 0.001-0.3, two orders
    of magnitude; other params sampled on their natural (linear/int) scale.
  Step 2 (efficient search + pruning): TPESampler (Bayesian — learns
    which regions of the space look promising from prior trials) +
    MedianPruner (kills a trial partway through CV if its running score
    is worse than the median of prior trials at the same fold index).
  Step 3 (robust CV, business metric): every trial is scored by k-fold
    CV average_precision (PR-AUC), never a single train/val split.
  Step 4 (early stopping): GradientBoostingClassifier's native
    n_iter_no_change/validation_fraction/tol — the model itself halts
    boosting once its internal validation score stops improving, so
    n_estimators_max is a ceiling, not a value every trial actually reaches.
"""
import logging
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score

log = logging.getLogger("src.tuning.search")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sample_params(trial, cfg) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", *cfg.max_depth_range),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.learning_rate_range, log=True),
        "subsample": trial.suggest_float("subsample", *cfg.subsample_range),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *cfg.min_samples_leaf_range),
    }


def _build_model(params: dict, cfg) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=cfg.n_estimators_max,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        min_samples_leaf=params["min_samples_leaf"],
        n_iter_no_change=cfg.early_stopping_rounds,
        validation_fraction=0.15,
        tol=1e-4,
        random_state=cfg.seed,
    )


def make_objective(X_train_p, y_train, cfg, trial_log: list):
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    def objective(trial) -> float:
        params = _sample_params(trial, cfg)
        fold_scores = []
        n_estimators_used = []

        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_train_p, y_train)):
            model = _build_model(params, cfg)
            model.fit(X_train_p.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = model.predict_proba(X_train_p.iloc[va_idx])[:, 1]
            score = average_precision_score(y_train.iloc[va_idx], proba)
            fold_scores.append(score)
            n_estimators_used.append(model.n_estimators_)

            running_mean = float(np.mean(fold_scores))
            trial.report(running_mean, step=fold_i)
            if trial.should_prune():
                trial_log.append({
                    "trial_number": trial.number, "params": params, "status": "PRUNED",
                    "folds_completed": fold_i + 1, "partial_mean_score": round(running_mean, 4),
                })
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        trial_log.append({
            "trial_number": trial.number, "params": params, "status": "COMPLETE",
            "folds_completed": cfg.cv_folds, "mean_cv_score": round(mean_score, 4),
            "std_cv_score": round(float(np.std(fold_scores)), 4),
            "mean_n_estimators_used": round(float(np.mean(n_estimators_used)), 1),
        })
        return mean_score

    return objective


def run_search(X_train_p, y_train, cfg):
    trial_log = []
    sampler = TPESampler(seed=cfg.seed)
    warmup = cfg.pruner_warmup_steps if cfg.pruner_warmup_steps < cfg.cv_folds else max(cfg.cv_folds - 2, 0)
    pruner = MedianPruner(n_warmup_steps=warmup)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(X_train_p, y_train, cfg, trial_log)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    n_pruned = sum(1 for t in trial_log if t["status"] == "PRUNED")
    n_complete = sum(1 for t in trial_log if t["status"] == "COMPLETE")
    if n_complete == 0:
        log.warning("[Step 2/3] Search complete: %s trials, 0 completed (%s pruned) — "
                     "no valid best trial to report.", len(trial_log), n_pruned)
    else:
        log.info("[Step 2/3] Search complete: %s trials (%s completed, %s pruned early -- compute saved), "
                  "best CV score=%.4f, best params=%s",
                  len(trial_log), n_complete, n_pruned, study.best_value, study.best_params)
    return study, trial_log
