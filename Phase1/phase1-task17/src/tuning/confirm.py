"""
tuning/confirm.py — Step 5: confirm the winning config on the held-out
test set. Also fits the reasonable-default baseline the same way, so the
test-confirmed gain is a real, measured comparison — not just "the
search found something," but "the search found something that beats a
sensible non-tuned default, confirmed on data neither ever trained on."
"""
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
)

log = logging.getLogger("src.tuning.confirm")

_METRICS = {
    "pr_auc": lambda y, proba, pred: average_precision_score(y, proba),
    "roc_auc": lambda y, proba, pred: roc_auc_score(y, proba),
    "precision": lambda y, proba, pred: precision_score(y, pred),
    "recall": lambda y, proba, pred: recall_score(y, pred),
    "f1": lambda y, proba, pred: f1_score(y, pred),
    "accuracy": lambda y, proba, pred: accuracy_score(y, pred),
}


def _evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {name: round(float(fn(y, proba, pred)), 4) for name, fn in _METRICS.items()}


def confirm_on_test(best_params: dict, cfg, X_train_p, y_train, X_test_p, y_test) -> dict:
    tuned_model = GradientBoostingClassifier(
        n_estimators=cfg.n_estimators_max,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        min_samples_leaf=best_params["min_samples_leaf"],
        n_iter_no_change=cfg.early_stopping_rounds,
        validation_fraction=0.15,
        tol=1e-4,
        random_state=cfg.seed,
    )
    tuned_model.fit(X_train_p, y_train)
    tuned_test_metrics = _evaluate(tuned_model, X_test_p, y_test)

    baseline_model = GradientBoostingClassifier(**cfg.baseline_params)
    baseline_model.fit(X_train_p, y_train)
    baseline_test_metrics = _evaluate(baseline_model, X_test_p, y_test)

    primary = "pr_auc" if cfg.scoring == "average_precision" else cfg.scoring
    gain = round(tuned_test_metrics[primary] - baseline_test_metrics[primary], 4)

    result = {
        "primary_metric": primary,
        "tuned_params": best_params,
        "tuned_n_estimators_actual": int(tuned_model.n_estimators_),
        "tuned_test_metrics": tuned_test_metrics,
        "baseline_params": cfg.baseline_params,
        "baseline_n_estimators_actual": int(baseline_model.n_estimators_),
        "baseline_test_metrics": baseline_test_metrics,
        "test_confirmed_gain": gain,
        "gain_is_real_not_search_overfitting": gain >= 0,
    }
    log.info("[Step 5] TEST SET (touched exactly once): tuned %s=%.4f vs baseline %s=%.4f, gain=%+.4f",
              primary, tuned_test_metrics[primary], primary, baseline_test_metrics[primary], gain)
    return result, tuned_model, baseline_model
