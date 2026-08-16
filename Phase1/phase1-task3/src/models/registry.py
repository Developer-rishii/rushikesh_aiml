"""
model registry — the ONE place to add a new model.

To add a new model:
  1. Write a factory function `def build_my_model(params: dict): -> estimator`
     that returns any scikit-learn-compatible estimator (has .fit/.predict/
     .predict_proba).
  2. Register it in REGISTRY below with a short string key.
  3. Set `model.name: "<that key>"` in configs/config.yaml.
That's it — nothing in dataset.py, build.py, harness.py, or evaluate.py
needs to change. This is what makes the skeleton "swap models without
rewrites."
"""
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def build_logreg_baseline(params: dict):
    return LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    )


def build_dummy_baseline(params: dict):
    return DummyClassifier(strategy=params.get("strategy", "most_frequent"))


def build_random_forest(params: dict):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", None),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    )


REGISTRY = {
    "logreg_baseline": build_logreg_baseline,
    "dummy_baseline": build_dummy_baseline,
    "random_forest": build_random_forest,
}


def build_model(name: str, params: dict):
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}. "
            f"Add a new one in src/models/registry.py."
        )
    return REGISTRY[name](params)
