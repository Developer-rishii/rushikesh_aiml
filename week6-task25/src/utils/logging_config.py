"""
utils/logging_config.py

One place to configure logging so every stage (data gen, training,
monitoring, API) writes consistent, timestamped lines to both the console
and evidence/run_logs.txt - that log file itself is submitted as live-run
evidence (Section 8: "demoed live; real numbers, not claims").
"""

import logging
import os

from src.config import EVIDENCE_DIR

LOG_PATH = os.path.join(EVIDENCE_DIR, "run_logs.txt")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
