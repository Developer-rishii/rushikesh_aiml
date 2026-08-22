"""
models/base.py — Step 1: build a few diverse base models, each wrapped
in the same preprocessing so every base model and the ensemble see
identically prepared data (no train/serve drift between members).
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

MODEL_BUILDERS = {
    "logreg": lambda params: LogisticRegression(**params),
    "gradient_boosting": lambda params: GradientBoostingClassifier(**params),
    "random_forest": lambda params: RandomForestClassifier(**params),
}


def _preprocessing_steps(cfg):
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    return steps


def build_base_pipeline(cfg, model_name: str) -> Pipeline:
    if model_name not in cfg.base_models:
        raise ValueError(f"Unknown base model '{model_name}'. Configured: {list(cfg.base_models.keys())}")
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"No builder registered for '{model_name}'. Available: {list(MODEL_BUILDERS.keys())}")
    params = cfg.base_models[model_name]
    steps = _preprocessing_steps(cfg) + [("model", MODEL_BUILDERS[model_name](params))]
    return Pipeline(steps)


def build_all_base_pipelines(cfg) -> dict:
    return {name: build_base_pipeline(cfg, name) for name in cfg.base_models}
