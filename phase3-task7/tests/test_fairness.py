import pandas as pd
from src.cold_start_recommender import ColdStartRecommender
from src.fairness import audit_fairness

jobs = pd.read_json("data/jobs.json")
users = pd.read_json("data/users.json")
cold_users = users[users["is_cold_start"]]
recommender = ColdStartRecommender()

def test_fairness_audit_returns_valid_metrics():
    report = audit_fairness(cold_users, jobs, recommender)
    assert "demographic_parity_ratio" in report
    assert "passed_80pct_rule" in report
    assert report["demographic_parity_ratio"] > 0.0
    assert report["passed_80pct_rule"] is True

def test_fairness_audit_empty_input():
    empty_users = cold_users.iloc[0:0]
    report = audit_fairness(empty_users, jobs, recommender)
    assert report["status"] == "EMPTY_INPUT"
    assert report["passed"] is True
