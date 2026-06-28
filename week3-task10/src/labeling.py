"""
PlaceMux Quality Sign-Off - Labeling Rule
===========================================
Explicit, auditable labeling rule for training the binary classifier.
"""

import numpy as np
import pandas as pd

def _parse_required_levels(row: pd.Series) -> dict[str, int]:
    raw = row.get("required_levels", "")
    if pd.isna(raw) or raw == "":
        return {}
    pairs = {}
    for chunk in str(raw).split("|"):
        if ":" in chunk:
            sk, lv = chunk.rsplit(":", 1)
            try:
                pairs[sk.strip()] = int(lv)
            except ValueError:
                pass
    return pairs


def compute_label(student: pd.Series, job: pd.Series) -> int:
    """Apply the labeling rule. Returns 1 (good match) or 0."""
    req_levels = _parse_required_levels(job)
    if not req_levels:
        return 0  # no required skills -> not a meaningful match

    total_required = len(req_levels)
    covered = 0
    level_ok = True

    for sk, req_lv in req_levels.items():
        stu_lv = student.get(sk, np.nan)
        if pd.notna(stu_lv) and stu_lv > 0:
            covered += 1
            if stu_lv < req_lv - 1:
                level_ok = False

    coverage_ratio = covered / total_required
    if coverage_ratio >= 0.80 and level_ok:
        return 1
    return 0


def label_dataset(students: pd.DataFrame, jobs: pd.DataFrame,
                  events: pd.DataFrame) -> pd.Series:
    """Compute binary labels for all events. Returns a Series aligned to events index."""
    stu_map = students.set_index("student_id")
    job_map = jobs.set_index("job_id")

    labels = []
    for _, ev in events.iterrows():
        sid, jid = ev["student_id"], ev["job_id"]
        if sid not in stu_map.index or jid not in job_map.index:
            labels.append(0)
            continue
        labels.append(compute_label(stu_map.loc[sid], job_map.loc[jid]))

    return pd.Series(labels, name="is_good_match")
