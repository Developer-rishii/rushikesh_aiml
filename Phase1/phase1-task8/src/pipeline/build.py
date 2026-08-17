"""
pipeline/build.py — Step 1: chain preprocessing + model into a SINGLE
sklearn Pipeline object. This is the direct, structural answer to the
brief's #1 pitfall ("Preprocessing applied outside the pipeline"): there
is no code path anywhere in this project that calls `.fit_transform()`
on the preprocessor separately from `.fit()` on the model — they are one
`Pipeline` object, fit with one `.fit()` call, saved with one
`joblib.dump()`, and reused with one `.predict()`/`.predict_proba()`
call. Preprocessing physically cannot travel separately from the model
because they are not separate objects.
"""
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger("src.pipeline.build")

MODEL_REGISTRY = {
    "logreg": lambda params: LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    ),
    "decision_tree": lambda params: DecisionTreeClassifier(
        max_depth=params.get("max_depth", None),
        class_weight=params.get("class_weight", None),
        random_state=params.get("random_state", 42),
    ),
}


def build_pipeline(cfg) -> Pipeline:
    """Build (unfitted) the single end-to-end Pipeline: impute -> scale -> model.
    All locked features here are numeric (Task 7's output), so this is a
    plain Pipeline rather than a ColumnTransformer — still ONE object."""
    if cfg.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{cfg.model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODEL_REGISTRY[cfg.model_name](cfg.model_params)))

    pipeline = Pipeline(steps)
    log.info("Built single sklearn Pipeline with steps: %s", [s[0] for s in steps])
    return pipeline
