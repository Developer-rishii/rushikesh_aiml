"""
Shared configuration for the latency-profiling & bottleneck-elimination project.
Single source of truth so every script (train, serve, evaluate) agrees on
paths, feature list, and the latency SLO we are being held to.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
EXP_DIR = os.path.join(ROOT, "experiments")
MODEL_DIR = os.path.join(EXP_DIR, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

for d in (DATA_DIR, EXP_DIR, MODEL_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

INTERACTIONS_CSV = os.path.join(DATA_DIR, "interaction_logs.csv")
EXPERIMENT_LOG = os.path.join(EXP_DIR, "experiment_log.jsonl")

RANDOM_SEED = 42

# Numeric + categorical features computed for every (job, candidate) pair.
# These are deliberately split into "cheap" (already on the request) and
# "fetched" (require a feature-store / DB round trip) so the profiler can
# show where time actually goes.
CHEAP_FEATURES = [
    "skill_overlap", "years_experience", "distance_km", "salary_gap_pct",
]
FETCHED_FEATURES = [
    "candidate_activity_score", "candidate_past_response_rate",
    "job_fill_urgency", "recruiter_rating",
]
ALL_FEATURES = CHEAP_FEATURES + FETCHED_FEATURES
PROTECTED_GROUP_COL = "candidate_group"  # synthetic attribute for the fairness sanity check
LABEL_COL = "relevance"  # 0 = no action, 1 = clicked, 2 = applied/shortlisted

# Latency SLO this task must hit (Stage C bar). Chosen to reflect a
# realistic p95 budget for an in-request ranking call.
LATENCY_P95_SLO_MS = 40.0

# Simulated per-call cost, used to turn latency into a $ number for the
# before/after cost comparison (Stage D deliverable). Modeled as linear in
# compute-ms, which is how most serverless / autoscaled inference is billed.
COST_PER_MS_USD = 0.0000009  # ~ $0.9 per million compute-ms, illustrative
REQUESTS_PER_DAY = 2_000_000  # marketplace-scale traffic assumption, stated explicitly
