"""
models/ensemble.py — Step 2: combine base models via voting (soft,
bagging-style averaging) and stacking (a learned meta-model).

Leakage guard for stacking (pitfall: "Stacking that leaks across folds"):
sklearn's StackingClassifier internally cross-validates each base
estimator to generate the meta-features it trains the final estimator
on — a base model's prediction used to train the meta-model on row i is
NEVER produced by a copy of that base model that was itself trained on
row i. This is built into StackingClassifier's `cv` parameter, not
something we have to hand-roll, but it's verified explicitly in
tests/test_ensemble.py by checking predictions differ from a
naively-leaky version.
"""
import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.base import build_all_base_pipelines

log = logging.getLogger("src.models.ensemble")

META_MODEL_BUILDERS = {
    "logreg": lambda: LogisticRegression(max_iter=2000),
}


def build_voting_ensemble(cfg):
    base_pipelines = build_all_base_pipelines(cfg)
    estimators = [(name, pipe) for name, pipe in base_pipelines.items()]
    ensemble = VotingClassifier(estimators=estimators, voting=cfg.voting_type)
    log.info("[Step 2] Built VOTING ensemble (%s) from base models: %s",
              cfg.voting_type, list(base_pipelines.keys()))
    return ensemble


def build_stacking_ensemble(cfg):
    if cfg.stacking_meta_model not in META_MODEL_BUILDERS:
        raise ValueError(f"Unknown meta-model '{cfg.stacking_meta_model}'. "
                          f"Available: {list(META_MODEL_BUILDERS.keys())}")
    base_pipelines = build_all_base_pipelines(cfg)
    estimators = [(name, pipe) for name, pipe in base_pipelines.items()]
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=META_MODEL_BUILDERS[cfg.stacking_meta_model](),
        cv=cfg.stacking_cv_folds,   # <-- the leakage guard: internal CV for meta-features
        passthrough=False,
    )
    log.info("[Step 2] Built STACKING ensemble (meta=%s, internal cv=%s folds) from base models: %s",
              cfg.stacking_meta_model, cfg.stacking_cv_folds, list(base_pipelines.keys()))
    return ensemble
