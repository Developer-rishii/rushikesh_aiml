"""
modeling/registry.py — Step 2: the first real model, and the one place to
add more later (continuing the pattern from Task 3's registry).
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_logreg(params: dict):
    return LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    )


def build_decision_tree(params: dict):
    return DecisionTreeClassifier(
        max_depth=params.get("max_depth", None),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    )


REGISTRY = {
    "logreg": build_logreg,
    "decision_tree": build_decision_tree,
}


def build_model(name: str, params: dict):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}.")
    return REGISTRY[name](params)
