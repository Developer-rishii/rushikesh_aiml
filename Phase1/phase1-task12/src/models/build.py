"""models/build.py — Step 1: the base classifier, single Pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MODEL_BUILDERS = {"logreg": lambda params: LogisticRegression(**params)}


def build_base_pipeline(cfg) -> Pipeline:
    if cfg.model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{cfg.model_name}'. Available: {list(MODEL_BUILDERS.keys())}")
    steps = [("impute", SimpleImputer(strategy=cfg.numeric_impute_strategy))]
    if cfg.scale_numeric:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODEL_BUILDERS[cfg.model_name](cfg.model_params)))
    return Pipeline(steps)
