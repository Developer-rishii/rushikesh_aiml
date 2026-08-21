"""
models/build.py — Step 2: build the linear baseline (Task 9's confirmed
config) and a more expressive gradient boosting model, both as single
sklearn Pipeline objects (same discipline as Tasks 8-9).
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def _preprocessing_steps(cfg):
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    return steps


def build_baseline_pipeline(cfg) -> Pipeline:
    steps = _preprocessing_steps(cfg)
    steps.append(("model", LogisticRegression(**cfg.baseline_model_params)))
    return Pipeline(steps)


def build_nonlinear_pipeline(cfg, model_params: dict = None) -> Pipeline:
    if cfg.nonlinear_model_name != "gradient_boosting":
        raise ValueError(f"Unknown nonlinear model '{cfg.nonlinear_model_name}'. "
                          f"Only 'gradient_boosting' is implemented.")
    params = model_params or {}
    steps = _preprocessing_steps(cfg)
    steps.append(("model", GradientBoostingClassifier(random_state=cfg.seed, **params)))
    return Pipeline(steps)
