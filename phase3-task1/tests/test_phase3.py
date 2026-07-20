"""
tests/test_phase3.py — pytest suite for Task 01 Phase 3
Run: pytest tests/ -v
All 14 tests must pass before submission.
"""

import sys, json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from health_monitor import (compute_health_report, ndcg_at_k,
                              _validate_inputs as validate_health_inputs)
from defect_ranker  import (build_features, score_one, validate_defect_labels,
                              MODEL_PATH)
from backlog_generator import generate_backlog


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def pred():
    return pd.read_csv(ROOT / "data" / "prediction_logs.csv")

@pytest.fixture(scope="module")
def inter():
    return pd.read_csv(ROOT / "data" / "interaction_logs.csv")

@pytest.fixture(scope="module")
def health():
    return json.loads((ROOT / "reports" / "health_report.json").read_text())

@pytest.fixture(scope="module")
def backlog():
    return json.loads((ROOT / "reports" / "phase3_backlog.json").read_text())


# ── DELIVERABLE 1: Health report ───────────────────────────────────────────────

def test_health_report_saved():
    """Health report JSON must exist and contain required keys."""
    p = ROOT / "reports" / "health_report.json"
    assert p.exists()
    r = json.loads(p.read_text())
    assert "summary" in r
    assert "by_model_version" in r
    assert "train_serve_skew" in r
    assert "by_segment" in r


def test_online_offline_gap_exists(health):
    """Online/offline gap must be computed and reported."""
    s = health["summary"]
    assert "online_offline_gap" in s
    assert "ndcg_at_5_offline" in s
    assert "ctr_online" in s
    assert isinstance(s["online_offline_gap"], float)


def test_skew_detected_in_at_least_one_feature(health):
    """Train/serve skew must be detected in at least one feature (we injected it)."""
    skewed = [f for f, v in health["train_serve_skew"].items() if v["skew_detected"]]
    assert len(skewed) >= 1, f"Expected skew detection, found none. Results: {health['train_serve_skew']}"


def test_per_version_breakdown(health):
    """Health report must have per-model-version breakdown."""
    assert len(health["by_model_version"]) >= 2
    for v, stats in health["by_model_version"].items():
        assert "ndcg_at_5" in stats
        assert "ctr" in stats
        assert "online_offline_gap" in stats


def test_validate_health_inputs_missing_cols():
    """DEPENDENCY: missing columns raises ValueError, not a silent proceed."""
    bad = pd.DataFrame({"log_id": ["L001"]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_health_inputs(bad, bad, bad, bad)


def test_ndcg_at_k_correctness():
    """nDCG@5 should be 1.0 for a perfectly ranked result."""
    df = pd.DataFrame({
        "student_id":  ["S001", "S001", "S001"],
        "served_score": [0.9, 0.6, 0.3],
        "clicked":      [1, 0, 0],
    })
    score = ndcg_at_k(df, k=5, score_col="served_score", relevance_col="clicked")
    assert score == pytest.approx(1.0, abs=0.01)


# ── DELIVERABLE 2: Defect ranker ───────────────────────────────────────────────

def test_model_artifact_exists():
    """Defect classifier model artifact must exist after training."""
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"


def test_ranked_defects_saved():
    """ranked_defects.csv must exist with expected columns."""
    p = ROOT / "data" / "ranked_defects.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert "defect_probability" in df.columns
    assert "estimated_user_impact" in df.columns
    assert "defect_rank" in df.columns
    assert "defect_category" in df.columns


def test_defects_ranked_by_impact():
    """Defects must be sorted by estimated_user_impact descending."""
    df = pd.read_csv(ROOT / "data" / "ranked_defects.csv")
    impacts = df["estimated_user_impact"].tolist()
    assert impacts == sorted(impacts, reverse=True), "Defects not sorted by impact"


def test_score_one_returns_reason(pred):
    """Every scored log must return a non-empty reason string."""
    log_id = pred["log_id"].iloc[0]
    result = score_one(log_id, pred)
    assert "reason" in result
    assert len(result["reason"]) > 20
    assert result["verdict"] in ("⚠️ DEFECT", "✅ OK")


def test_score_one_unknown_log(pred):
    """Unknown log_id returns error dict, not an exception."""
    result = score_one("L_UNKNOWN_999", pred)
    assert "error" in result


def test_validate_defect_labels_missing_cols():
    """DEPENDENCY: defect_labels missing columns raises ValueError."""
    bad = pd.DataFrame({"log_id": ["L001"], "is_defect": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_defect_labels(bad)


# ── DELIVERABLE 3: Backlog ─────────────────────────────────────────────────────

def test_backlog_saved():
    """Phase-3 backlog JSON must exist."""
    p = ROOT / "reports" / "phase3_backlog.json"
    assert p.exists()
    bl = json.loads(p.read_text())
    assert "items" in bl
    assert len(bl["items"]) >= 3


def test_backlog_ranked_by_priority(backlog):
    """P0 items must appear before P1 items in the backlog."""
    priorities = [i["priority"] for i in backlog["items"]]
    p0_indices = [i for i, p in enumerate(priorities) if p == "P0"]
    p1_indices = [i for i, p in enumerate(priorities) if p == "P1"]
    if p0_indices and p1_indices:
        assert max(p0_indices) < min(p1_indices), "P0 items must precede P1 items"


def test_backlog_items_have_evidence(backlog):
    """Every backlog item must have non-empty evidence and a metric to move."""
    for item in backlog["items"]:
        assert len(item.get("evidence", "")) > 20, f"{item['backlog_id']} lacks evidence"
        assert len(item.get("metric_to_move", "")) > 5, f"{item['backlog_id']} missing metric"


def test_experiment_log_written():
    """Experiment log must exist with model metrics."""
    p = ROOT / "reports" / "experiment_log.jsonl"
    assert p.exists()
    entries = [json.loads(l) for l in p.read_text().strip().splitlines() if l.strip()]
    assert len(entries) >= 1
    last = entries[-1]
    assert "precision" in last
    assert "f1" in last
    assert "n_defects_in_all_logs" in last
