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

        fs = raw["feature_selection"]
        self.correlation_threshold = fs["correlation_threshold"]
        self.drop_near_zero_variance = fs["drop_near_zero_variance"]
        self.variance_threshold = fs["variance_threshold"]

        sc = raw["scaling"]
        self.scaling_method = sc["method"]

        p = raw["pca"]
        self.pca_apply = p["apply"]
        self.pca_variance_to_retain = p["variance_to_retain"]
        self.pca_max_components = p["max_components"]

        k = raw["k_selection"]
        self.k_range = k["k_range"]
        self.k_n_init = k["n_init"]

        s = raw["sanity_check"]
        self.min_acceptable_silhouette = s["min_acceptable_silhouette"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.prepared_data_dir = ROOT_DIR / lg["prepared_data_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(seed={self.seed})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "feature_selection", "scaling", "pca", "k_selection",
                "sanity_check", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
