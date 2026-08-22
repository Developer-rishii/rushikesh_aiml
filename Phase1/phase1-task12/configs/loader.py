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

        m = raw["model"]
        self.model_name = m["name"]
        self.model_params = m.get("params", {})

        c = raw["calibration"]
        self.calibration_methods = c["methods"]
        self.calibration_cv_folds = c["cv_folds"]
        self.n_calibration_bins = c["n_calibration_bins"]

        co = raw["cost"]
        self.cost_false_negative = co["cost_false_negative"]
        self.cost_false_positive = co["cost_false_positive"]

        s = raw["segments"]
        self.segment_feature = s["segment_feature"]
        self.n_segments = s["n_segments"]
        self.segment_labels = s["segment_labels"]

        e = raw["evaluation"]
        self.eval_cv_folds = e["cv_folds"]
        self.primary_metric = e["primary_metric"]
        self.metrics = e["metrics"]
        self.min_segment_recall = e["min_segment_recall"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.serving_dir = ROOT_DIR / lg["serving_dir"]

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __repr__(self):
        return f"Config(model={self.model_name}, seed={self.seed})"


def load_config(config_path: Path = None) -> Config:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    required = {"seed", "data", "preprocessing", "model", "calibration", "cost",
                "segments", "evaluation", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
