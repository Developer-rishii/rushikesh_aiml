"""
Central, single-source-of-truth config for Task 2.
Continuity with Task 1: same SEED, same dataset, same TARGET_COL name,
so results from both tasks are directly comparable.
"""
from pathlib import Path

# ---- Reproducibility (matches Task 1) ---------------------------------
SEED = 42

# ---- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw_from_task1.csv"      # carried over from Task 1
ENRICHED_PATH = DATA_DIR / "raw_enriched.csv"    # + demo leakage/id columns
CLEAN_PATH = DATA_DIR / "clean.csv"              # post leakage-removal
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
LOGS_DIR = ROOT_DIR / "outputs" / "logs"

# ---- Target / problem definition (Step 1-2 of the build pipeline) -------
# Same target as Task 1: Wisconsin Diagnostic Breast Cancer, malignant(0)
# vs benign(1) — Task 1 proved the env + split + baseline works; Task 2
# scopes the feature set properly before further modelling.
TARGET_COL = "target"
PROBLEM_TYPE = "binary_classification"
SUCCESS_METRIC = "PR-AUC"
PROBLEM_STATEMENT = (
    "Predict whether a breast tumor is malignant or benign from digitized "
    "fine-needle-aspirate (FNA) image measurements, available at the time "
    "of biopsy analysis (no post-diagnosis information)."
)

# ---- Split ratios (matches Task 1) --------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
