"""
Study guide, Stage A: "Confirm every prerequisite and upstream input above
is actually in hand" and "Waiting on: Audit results. If it's late, your
task slips -- chase it early..."

This module is that check, made real: Task 24 does not proceed past this
point until the Task 21 hand-off artifact is present, parseable, and
contains the fields this task depends on.
"""
import json
import os


class DependencyError(Exception):
    """Raised when an upstream hand-off artifact is missing or malformed."""


REQUIRED_TASK21_FIELDS = {
    "disparate_impact", "group_positive_rates", "finding", "protected_attribute",
}


def load_task21_audit(path: str) -> dict:
    if not os.path.exists(path):
        raise DependencyError(
            f"Upstream dependency missing: Task 21 audit results not found at '{path}'. "
            "Per the study guide: 'If it's late, your task slips -- chase it early and "
            "agree an ETA rather than waiting silently.' Run task21_audit.py first."
        )
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DependencyError(
            f"Upstream dependency at '{path}' is not valid JSON ({e}). "
            "Cannot safely proceed with a corrupted hand-off artifact."
        ) from e

    missing = REQUIRED_TASK21_FIELDS - set(data.keys())
    if missing:
        raise DependencyError(
            f"Upstream dependency at '{path}' is missing required fields: {sorted(missing)}. "
            "The hand-off contract has changed or the file is incomplete."
        )
    return data
