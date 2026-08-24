"""
validation/kfold_compare.py — Steps 1-3, 5: choose stratified CV, run
K-Fold collecting PER-FOLD scores for every candidate, on the IDENTICAL
fold split (same StratifiedKFold object reused, not re-instantiated per
model — this is what makes the comparison fair/apples-to-apples), and
report mean AND spread, never just the best fold or a single number.
"""
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.validation.models import build_pipeline

log = logging.getLogger("src.validation.kfold_compare")


def compare_candidates_cv(X, y, cfg) -> dict:
    cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.seed)

    results = {}
    for model_name in cfg.candidate_models:
        pipeline = build_pipeline(cfg, model_name)
        fold_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=cfg.scoring, n_jobs=-1)
        results[model_name] = {
            "fold_scores": [round(float(s), 4) for s in fold_scores],
            "n_folds": cfg.outer_folds,
            "mean": round(float(fold_scores.mean()), 4),
            "std": round(float(fold_scores.std()), 4),
            "min": round(float(fold_scores.min()), 4),
            "max": round(float(fold_scores.max()), 4),
        }
        log.info("[Step 2/3] %s: fold_scores=%s mean=%.4f std=%.4f (spread reported, not just best fold)",
                  model_name, results[model_name]["fold_scores"], results[model_name]["mean"],
                  results[model_name]["std"])

    class_balance_per_fold = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        rate = float(y.iloc[test_idx].mean())
        class_balance_per_fold.append(round(rate, 4))
    overall_rate = float(y.mean())

    return {
        "cv_scheme": "StratifiedKFold",
        "outer_folds": cfg.outer_folds,
        "scoring": cfg.scoring,
        "per_model_results": results,
        "stratification_check": {
            "overall_positive_rate": round(overall_rate, 4),
            "per_fold_positive_rate": class_balance_per_fold,
            "max_deviation_from_overall": round(max(abs(r - overall_rate) for r in class_balance_per_fold), 4),
        },
    }
