import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
