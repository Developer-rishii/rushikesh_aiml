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
        self.target_col = d["target_col"]
        self.train_frac = d["train_frac"]
        self.val_frac = d["val_frac"]
        self.test_frac = d["test_frac"]

        p = raw["preprocessing"]
        self.numeric_impute_strategy = p["numeric_impute_strategy"]
        self.scale_numeric = p["scale_numeric"]

        m = raw["model"]
        self.model_name = m["name"]
        self.model_params = m.get("params", {})

        pr = raw["pruning"]
        self.min_importance_lift = pr["min_importance_lift"]
        self.leakage_corr_threshold = pr["leakage_corr_threshold"]

        e = raw["evaluation"]
        self.primary_metric = e["primary_metric"]
        self.n_permutation_repeats = e["n_permutation_repeats"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.baseline_feature_dir = ROOT_DIR / lg["baseline_feature_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(model={self.model_name}, seed={self.seed}, source={self.config_path})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "preprocessing", "model", "pruning", "evaluation", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
