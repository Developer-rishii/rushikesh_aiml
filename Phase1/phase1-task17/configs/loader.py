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
        self.test_frac = d["test_frac"]

        p = raw["preprocessing"]
        self.numeric_impute_strategy = p["numeric_impute_strategy"]
        self.scale_numeric = p["scale_numeric"]

        s = raw["search"]
        self.n_estimators_max = s["n_estimators_max"]
        self.max_depth_range = s["max_depth"]
        self.learning_rate_range = s["learning_rate"]
        self.subsample_range = s["subsample"]
        self.min_samples_leaf_range = s["min_samples_leaf"]
        self.n_trials = s["n_trials"]
        self.cv_folds = s["cv_folds"]
        self.scoring = s["scoring"]
        self.early_stopping_rounds = s["early_stopping_rounds"]
        self.pruner_warmup_steps = s["pruner_warmup_steps"]

        self.baseline_params = raw["baseline"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.artifact_dir = ROOT_DIR / lg["artifact_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(n_trials={self.n_trials}, seed={self.seed})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "preprocessing", "search", "baseline", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
