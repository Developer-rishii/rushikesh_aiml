"""
Step 4 of the build pipeline: a simple experiment log so runs are
comparable (params + metrics), per-run, append-only CSV.
"""
import sys
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import EXPERIMENTS_LOG, EXPERIMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("experiment_tracker")


def log_run(run_name: str, params: dict, metrics: dict) -> None:
    """Append one row: timestamp, run_name, flattened params, flattened metrics."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        **{f"param_{k}": v for k, v in params.items()},
        **{f"metric_{k}": v for k, v in metrics.items()},
    }

    file_exists = EXPERIMENTS_LOG.exists()
    existing_fieldnames = []
    if file_exists:
        with open(EXPERIMENTS_LOG, "r", newline="") as f:
            reader = csv.reader(f)
            existing_fieldnames = next(reader, [])

    fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))

    with open(EXPERIMENTS_LOG, "w" if not file_exists else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    log.info("Logged run '%s' -> %s", run_name, EXPERIMENTS_LOG)
