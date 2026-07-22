import pandas as pd
from src.skew_check import check_skew

jobs = pd.read_json("data/jobs.json")
users = pd.read_json("data/users.json")

def test_train_serve_skew_zero():
    sample_skills = users["skills"].head(20).tolist()
    res = check_skew(sample_skills, jobs)
    assert res["status"] == "PASSED_ZERO_SKEW"
    assert res["max_diff"] < 1e-9
