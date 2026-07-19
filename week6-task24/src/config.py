"""Central configuration for the Task 24 fairness close + sign-off pipeline."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.csv")
TASK21_AUDIT_PATH = os.path.join(ARTIFACTS_DIR, "task21_audit_results.json")   # upstream dependency
ML_GO_AHEAD_PATH = os.path.join(ARTIFACTS_DIR, "ml_go_ahead.json")             # this task's hand-off
METRICS_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "metrics_report.json")

# Scale: real-sample-at-scale, not a toy. ~50 concurrent job postings,
# ~40k candidate-job scoring events, with realistic missingness/duplication.
N_ROWS = 40_000
N_JOBS = 50
SEED = 42

DI_TARGET = 0.80          # the 4/5ths rule
CEILING_TOLERANCE = 0.95  # "essentially at the data's fairness ceiling"
PROTECTED_ATTR = "college_tier"
LABEL_COL = "historical_recommended"
MERIT_FEATURES = ["skill_score", "years_exp", "jd_match", "portfolio_score"]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
