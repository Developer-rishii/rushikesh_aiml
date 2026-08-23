"""Single source of truth for configs/config.yaml -> typed, resolved Config."""
from pathlib import Path
import json
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
        self.prepared_data_path = ROOT_DIR / d["prepared_data_path"]
        self.locked_params_path = ROOT_DIR / d["locked_params_path"]
        self.raw_data_path = ROOT_DIR / d["raw_data_path"]
        self.target_col = d["target_col"]

        km = raw["kmeans"]
        self.kmeans_n_init = km["n_init"]

        st = raw["stability"]
        self.stability_seeds = st["seeds_to_check"]
        self.min_acceptable_ari = st["min_acceptable_ari"]

        pr = raw["profiling"]
        self.top_n_defining_features = pr["top_n_defining_features"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.artifact_dir = ROOT_DIR / lg["artifact_dir"]

        if not self.locked_params_path.exists():
            raise FileNotFoundError(f"Task 14's locked params not found at {self.locked_params_path}.")
        locked = json.loads(self.locked_params_path.read_text())
        if "chosen_k" not in locked:
            raise ValueError(f"'chosen_k' missing from {self.locked_params_path}.")
        self.k = locked["chosen_k"]
        self.k_justification_silhouette = locked.get("chosen_k_silhouette")
        self.selected_features = locked.get("selected_features", [])

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(k={self.k} [from Task14 lock], seed={self.seed})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "kmeans", "stability", "profiling", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
