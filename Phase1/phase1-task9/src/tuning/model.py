"""
tuning/model.py — the single sklearn Pipeline (impute -> scale -> model),
same discipline as Task 8: preprocessing and model are one object, so
tuning the model's hyperparameters via GridSearchCV never risks
preprocessing drifting outside the pipeline.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODEL_REGISTRY = {
    "logreg": lambda params: LogisticRegression(**params),
}


def build_pipeline(cfg, model_params: dict) -> Pipeline:
    if cfg.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{cfg.model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODEL_REGISTRY[cfg.model_name](model_params)))
    return Pipeline(steps)
