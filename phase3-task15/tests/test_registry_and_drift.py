import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.registry import ModelRegistry
from src.baseline import SkillMatchBaseline, fairness_report
from src.drift import DriftDetector, psi


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "job_id": rng.integers(1, 5, n),
        "skill_match_score": rng.uniform(0, 1, n),
        "embedding_similarity": rng.uniform(0, 1, n),
        "experience_years": rng.uniform(0, 10, n),
        "location_match": rng.integers(0, 2, n),
        "recruiter_response_rate": rng.uniform(0, 1, n),
        "past_ctr": rng.uniform(0, 1, n),
        "gender": rng.choice(["M", "F"], n),
        "shortlisted": rng.integers(0, 2, n),
    })


def test_registry_register_promote_rollback(tmp_path):
    reg = ModelRegistry(db_path=str(tmp_path / "reg.db"), artifact_dir=str(tmp_path / "models"))
    model = SkillMatchBaseline()
    df = _toy_df()
    reg.register("v1", model, df, "hash123", {"ndcg@10": 0.5})
    reg.register("v2", model, df, "hash123", {"ndcg@10": 0.6}, parent_version="v1")

    reg.promote("v1")
    assert reg.current_production()[0] == "v1"
    reg.promote("v2")
    assert reg.current_production()[0] == "v2"
    reg.rollback("v1", reason="test rollback")
    assert reg.current_production()[0] == "v1"

    loaded = reg.load_model("v1")
    assert loaded is not None
    assert len(reg.list_versions()) == 2
    assert len(reg.audit_trail()) == 3  # promote, promote, rollback


def test_psi_detects_shifted_distribution():
    ref = np.random.default_rng(1).normal(0, 1, 2000)
    same = np.random.default_rng(2).normal(0, 1, 2000)
    shifted = np.random.default_rng(3).normal(2.5, 1, 2000)
    assert psi(ref, same) < 0.10
    assert psi(ref, shifted) > 0.25


def test_drift_detector_triggers_retrain_on_performance_drop():
    d = DriftDetector(feature_columns=["skill_match_score"], report_dir="/tmp/drift_test_reports")
    ref = _toy_df(seed=1)
    comp = _toy_df(seed=1)  # same distribution -> no input drift
    report = d.evaluate_and_log(ref, comp, baseline_metric=0.6, current_metric=0.4, batch_name="unit_test")
    assert report["performance_drift"]["retrain_triggered"] is True
    assert report["retrain_triggered"] is True


def test_fairness_report_shape():
    df = _toy_df()
    scores = np.random.default_rng(0).uniform(0, 1, len(df))
    rep = fairness_report(df, scores)
    assert "demographic_parity_diff" in rep
    assert 0 <= rep["demographic_parity_diff"] <= 1
