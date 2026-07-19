"""
Central configuration for the PlaceMux Live Model Monitoring system.
Task 25 - Week 6 - Phase 3 - Go-Live.

Single source of truth for paths, feature schema, and monitoring thresholds
so every module (training, inference, monitoring, api) agrees on the same
contract. Change values here, not inside individual modules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
MONITORING_STORE_DIR = os.path.join(ROOT_DIR, "monitoring_store")
EVIDENCE_DIR = os.path.join(ROOT_DIR, "evidence")
PLOTS_DIR = os.path.join(EVIDENCE_DIR, "plots")

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, EXPERIMENTS_DIR,
           MONITORING_STORE_DIR, EVIDENCE_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_HISTORY_PATH = os.path.join(DATA_RAW_DIR, "historical_matches.csv")
PROD_TRAFFIC_PATH = os.path.join(DATA_RAW_DIR, "production_traffic_sample.csv")
REFERENCE_DIST_PATH = os.path.join(DATA_PROCESSED_DIR, "reference_distribution.json")

MODEL_PATH = os.path.join(MODELS_DIR, "match_model.joblib")
BASELINE_METRICS_PATH = os.path.join(MODELS_DIR, "baseline_metrics.json")
EXPERIMENT_LOG_PATH = os.path.join(EXPERIMENTS_DIR, "experiment_log.csv")

MONITORING_DB_PATH = os.path.join(MONITORING_STORE_DIR, "monitoring.db")

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# These are the signals PlaceMux's matching model sees. Each is derived from
# upstream tasks (verified skill scores, JD parsing, interview eval, resume
# parsing) - task 25 does not invent them, it consumes them.
FEATURE_COLUMNS = [
    "skill_overlap_score",     # 0-1, verified skills vs JD required skills
    "years_experience",        # numeric, candidate's relevant experience
    "experience_gap",          # JD required years - candidate years (can be negative)
    "resume_parse_confidence", # 0-1, confidence of the resume/JD parser
    "interview_eval_score",    # 0-1, from Task 3 interview evaluation model
    "communication_score",     # 0-1, verified communication/soft-skill score
    "role_historical_hire_rate",  # 0-1, base rate of hires for this role/segment
]

TARGET_COLUMN = "is_successful_match"
ID_COLUMNS = ["match_id", "student_id", "job_id", "segment", "event_time"]

# ---------------------------------------------------------------------------
# Monitoring thresholds ("what good looks like", quantified)
# ---------------------------------------------------------------------------
# A metric drop of more than this fraction relative to the validated baseline
# run is treated as a live degradation and raises an alert.
METRIC_DEGRADATION_ALERT_THRESHOLD = 0.15   # 15% relative drop

# Population Stability Index thresholds (industry-standard bands)
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

# Minimum labeled samples required in a monitoring window before we trust the
# computed precision/recall/FPR numbers (avoids noisy alerts on tiny samples).
MIN_LABELED_SAMPLES_FOR_METRICS = 30

# Rolling monitoring window size (number of production events per window)
MONITORING_WINDOW_SIZE = 200

# Fraction of live events whose ground-truth outcome ("was this actually a
# good hire") has arrived by the time we monitor them - hiring outcomes are
# not instant, so most systems only have partial labels at any moment.
LABEL_ARRIVAL_RATE = 0.55

RANDOM_SEED = 42
