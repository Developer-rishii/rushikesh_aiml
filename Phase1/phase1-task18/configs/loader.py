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
        self.corpus_path = ROOT_DIR / d["corpus_path"]
        self.text_col = d["text_col"]
        self.label_col = d["label_col"]
        self.test_frac = d["test_frac"]
        self.val_frac = d["val_frac"]

        p = raw["preprocessing"]
        self.lowercase = p["lowercase"]
        self.strip_punctuation = p["strip_punctuation"]
        self.remove_stopwords = p["remove_stopwords"]
        self.min_token_length = p["min_token_length"]

        v = raw["vectorization"]
        self.tfidf_ngram_range = tuple(v["tfidf"]["ngram_range"])
        self.tfidf_max_features = v["tfidf"]["max_features"]
        self.tfidf_min_df = v["tfidf"]["min_df"]
        self.lsa_n_components = v["lsa"]["n_components"]

        c = raw["classifier"]
        self.classifier_name = c["name"]
        self.classifier_params = c.get("params", {})

        e = raw["evaluation"]
        self.primary_metric = e["primary_metric"]
        self.n_errors_to_inspect = e["n_errors_to_inspect"]

        lg = raw["logging"]
        self.report_dir = ROOT_DIR / lg["report_dir"]
        self.figure_dir = ROOT_DIR / lg["figure_dir"]
        self.log_dir = ROOT_DIR / lg["log_dir"]
        self.artifact_dir = ROOT_DIR / lg["artifact_dir"]

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
    required = {"seed", "data", "preprocessing", "vectorization", "classifier", "evaluation", "logging"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {missing}")
    cfg = Config(raw, path)
    cfg.set_global_seed()
    return cfg
