"""
Single source of truth for turning configs/config.yaml into typed,
path-resolved access. Nothing else in this project reads the YAML
directly or hardcodes a path/param/seed.
"""
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

        en = raw["enrichment"]
        self.add_categorical_feature = en.get("add_categorical_feature", False)
        self.inject_missing_values = en.get("inject_missing_values", False)
        self.missing_fraction = en.get("missing_fraction", 0.0)

        p = raw["preprocessing"]
        self.numeric_impute_strategy = p["numeric_impute_strategy"]
        self.categorical_impute_strategy = p["categorical_impute_strategy"]
        self.scale_numeric = p["scale_numeric"]
        self.encode_categorical = p["encode_categorical"]

        b = raw["baseline"]
        self.baseline_strategy = b["strategy"]

        m = raw["model"]
        self.model_name = m["name"]
        self.model_params = m.get("params", {})

        e = raw["evaluation"]
        self.primary_metric = e["primary_metric"]
        self.metrics = e["metrics"]
        self.worst_errors_to_inspect = e.get("worst_errors_to_inspect", 10)

        lg = raw["logging"]
        self.experiment_log_path = ROOT_DIR / lg["experiment_log"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.model_dir = ROOT_DIR / lg["model_dir"]

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
    required = {"seed", "data", "enrichment", "preprocessing", "baseline", "model", "evaluation", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
