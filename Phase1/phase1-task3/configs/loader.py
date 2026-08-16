"""
Single source of truth for turning configs/config.yaml into something the
rest of the codebase uses. Nothing else in this project reads the YAML
file directly or hardcodes a path/param/seed -- everything goes through
load_config().
"""
from pathlib import Path
import random
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


class Config:
    """Thin typed wrapper around the parsed YAML dict with resolved paths."""

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

        f = raw["features"]
        self.drop_columns = f.get("drop_columns", [])
        self.scale = f.get("scale", True)
        self.impute_strategy = f.get("impute_strategy", "median")

        m = raw["model"]
        self.model_name = m["name"]
        self.model_params = m.get("params", {})

        e = raw["evaluation"]
        self.primary_metric = e["primary_metric"]
        self.metrics = e["metrics"]

        lg = raw["logging"]
        self.experiment_log_path = ROOT_DIR / lg["experiment_log"]
        self.run_log_dir = ROOT_DIR / lg["run_log_dir"]
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
    required_top = {"seed", "data", "features", "model", "evaluation", "logging"}
    missing = required_top - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
