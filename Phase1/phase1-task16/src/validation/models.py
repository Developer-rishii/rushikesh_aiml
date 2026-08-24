"""validation/models.py — single sklearn Pipeline builders, one per candidate."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

MODEL_BUILDERS = {
    "logreg": lambda p: LogisticRegression(**p),
    "random_forest": lambda p: RandomForestClassifier(**p),
    "gradient_boosting": lambda p: GradientBoostingClassifier(**p),
}


def build_pipeline(cfg, model_name: str, override_params: dict = None) -> Pipeline:
    if model_name not in cfg.candidate_models:
        raise ValueError(f"Unknown model '{model_name}'. Configured: {list(cfg.candidate_models.keys())}")
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"No builder for '{model_name}'. Available: {list(MODEL_BUILDERS.keys())}")
    params = override_params if override_params is not None else cfg.candidate_models[model_name]
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODEL_BUILDERS[model_name](params)))
    return Pipeline(steps)
