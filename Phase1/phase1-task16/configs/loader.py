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

        p = raw["preprocessing"]
        self.numeric_impute_strategy = p["numeric_impute_strategy"]
        self.scale_numeric = p["scale_numeric"]

        cv = raw["cv"]
        self.outer_folds = cv["outer_folds"]
        self.inner_folds = cv["inner_folds"]
        self.scoring = cv["scoring"]

        self.candidate_models = raw["candidate_models"]

        ncv = raw["nested_cv"]
        self.model_to_tune = ncv["model_to_tune"]
        self.nested_param_grid = ncv["param_grid"]

        sel = raw["selection"]
        self.max_acceptable_std = sel["max_acceptable_std"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(models={list(self.candidate_models.keys())}, seed={self.seed})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "preprocessing", "cv", "candidate_models", "nested_cv", "selection", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
