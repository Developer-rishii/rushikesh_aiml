"""
Central, portable path resolution. Every script imports ROOT from here
instead of hardcoding an absolute path - this is what makes the project
work regardless of where it's unzipped (previous version hardcoded
/home/claude/placemux_task14/... which broke the moment the folder was
renamed to phase3-task14 on a fresh machine - see reports/bugfix_log.md).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above src/
DATA_DIR = os.path.join(ROOT, "data")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
REPORTS_DIR = os.path.join(ROOT, "reports")

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
