"""
Step 6 entry point: run the full pipeline end-to-end.

    python -m src.run_pipeline

This is the single command that proves "everything works" — the Day 1
goal stated in the study guide. Each stage is wrapped so a failure at
any step names exactly which step and why, instead of a raw traceback.
"""
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import data_ingestion, data_split, smoke_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_pipeline")


def main():
    stages = [
        ("ingest", data_ingestion.ingest),
        ("split", data_split.run),
        ("smoke_test", smoke_test.run_smoke_test),
    ]

    for name, fn in stages:
        log.info("=== Stage: %s ===", name)
        try:
            fn()
        except Exception as exc:
            log.error("Pipeline FAILED at stage '%s': %s", name, exc)
            raise SystemExit(1) from exc

    log.info("Pipeline completed successfully end-to-end.")


if __name__ == "__main__":
    main()
