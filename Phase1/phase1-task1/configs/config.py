"""
Central, single-source-of-truth config for Task 1.
Every module imports SEED from here — never hardcode a seed elsewhere.
"""
from pathlib import Path

# ---- Reproducibility -------------------------------------------------
SEED = 42

# ---- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EXPERIMENTS_LOG = EXPERIMENTS_DIR / "experiment_log.csv"
MODEL_DIR = ROOT_DIR / "experiments" / "models"

# ---- Split ratios --------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

TARGET_COL = "target"
