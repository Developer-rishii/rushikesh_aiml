"""
tuning/confirm.py — Step 5: confirm the gain holds on the held-out test
set. This is the ONLY place X_test is used in this entire project run —
imported and called exactly once, after the search in search.py has
already completed and selected its winner. Both the default-params model
and the tuned model are refit on X_train (fair comparison, same data)
and each evaluated on X_test exactly once.
"""
import logging
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from src.tuning.model import build_pipeline

log = logging.getLogger("src.tuning.confirm")

_METRIC_FUNCS = {
    "pr_auc": lambda y, proba, pred: average_precision_score(y, proba),
    "roc_auc": lambda y, proba, pred: roc_auc_score(y, proba),
    "precision": lambda y, proba, pred: precision_score(y, pred),
    "recall": lambda y, proba, pred: recall_score(y, pred),
    "f1": lambda y, proba, pred: f1_score(y, pred),
    "accuracy": lambda y, proba, pred: accuracy_score(y, pred),
}


def _evaluate(pipeline, X, y) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return {name: round(float(fn(y, proba, pred)), 4) for name, fn in _METRIC_FUNCS.items()}


def confirm_test_gain(X_train, y_train, X_test, y_test, best_params: dict, cfg) -> dict:
    # strip the 'model__' prefix GridSearchCV requires back to plain kwargs
    tuned_model_params = {k.replace("model__", ""): v for k, v in best_params.items()
                           if k.startswith("model__")}
    full_tuned_params = {**cfg.default_params, **tuned_model_params}
    # tuned params should win on any overlapping key (e.g. penalty/C aren't
    # in default_params, but if they were, search wins)

    default_pipeline = build_pipeline(cfg, cfg.default_params)
    default_pipeline.fit(X_train, y_train)
    default_test_metrics = _evaluate(default_pipeline, X_test, y_test)

    tuned_pipeline = build_pipeline(cfg, full_tuned_params)
    tuned_pipeline.fit(X_train, y_train)
    tuned_test_metrics = _evaluate(tuned_pipeline, X_test, y_test)

    primary = "pr_auc" if cfg.scoring == "average_precision" else cfg.scoring
    gain = round(tuned_test_metrics[primary] - default_test_metrics[primary], 4)

    result = {
        "default_params": cfg.default_params,
        "tuned_params": full_tuned_params,
        "default_test_metrics": default_test_metrics,
        "tuned_test_metrics": tuned_test_metrics,
        "primary_metric": primary,
        "test_set_gain": gain,
        "gain_confirmed_on_test_set": gain >= 0,
    }
    log.info("[Step 5] TEST SET (touched exactly once): default %s=%.4f, tuned %s=%.4f, gain=%+.4f",
              primary, default_test_metrics[primary], primary, tuned_test_metrics[primary], gain)
    return result, tuned_pipeline, default_pipeline
