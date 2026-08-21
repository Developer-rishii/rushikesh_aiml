"""Single source of truth for configs/config.yaml -> typed, resolved Config."""
from pathlib import Path
import random
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


class Config:
    def __init__(self, raw: dict, config_path: Path):
        self._raw = raw
        self.config_path = config_path
        self.seed = raw["seed"]

        d = raw["data"]
        self.raw_data_path = ROOT_DIR / d["raw_path"]
        self.locked_features_path = ROOT_DIR / d["locked_features_path"]
        self.target_col = d["target_col"]
        self.train_frac = d["train_frac"]
        self.val_frac = d["val_frac"]
        self.test_frac = d["test_frac"]

        p = raw["preprocessing"]
        self.numeric_impute_strategy = p["numeric_impute_strategy"]
        self.scale_numeric = p["scale_numeric"]

        bm = raw["baseline_model"]
        self.baseline_model_name = bm["name"]
        self.baseline_model_params = bm.get("params", {})

        nm = raw["nonlinear_model"]
        self.nonlinear_model_name = nm["name"]
        self.nonlinear_search_space = nm["search_space"]
        self.nonlinear_cv_folds = nm["cv_folds"]
        self.nonlinear_scoring = nm["scoring"]

        e = raw["evaluation"]
        self.primary_metric = e["primary_metric"]
        self.metrics = e["metrics"]
        self.min_lift_to_keep = e["min_lift_to_keep"]

        pdp = raw["pdp"]
        self.pdp_n_top_features = pdp["n_top_features"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.artifact_dir = ROOT_DIR / lg["artifact_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(seed={self.seed}, source={self.config_path})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "preprocessing", "baseline_model", "nonlinear_model",
                "evaluation", "pdp", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
