"""
Automated pass/fail gate over the chaos engine results. Run with:
    pytest tests/test_chaos_scenarios.py -v
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = f"{BASE}/evidence/chaos_test_results.json"


def _run_chaos():
    subprocess.run([sys.executable, "chaos_engine.py"], cwd=f"{BASE}/src", check=True)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def test_all_scenarios_pass():
    results = _run_chaos()
    failed = [s["name"] for s in results["scenarios"] if not s["pass"]]
    assert not failed, f"Chaos scenarios failed: {failed}"


def test_model_down_pages_and_degrades():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    s = next(x for x in results["scenarios"] if x["name"] == "model_service_down")
    assert s["actual_mode"] == "heuristic"
    assert s["paged_on_call"] is True


def test_corrupted_features_never_crash():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    s = next(x for x in results["scenarios"] if x["name"] == "corrupted_features")
    assert s["no_nan_crash"] is True
    assert s["actual_mode"] == "heuristic"


def test_stale_features_trigger_degradation():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    s = next(x for x in results["scenarios"] if x["name"] == "stale_features")
    assert s["actual_mode"] == "heuristic"
    assert s["paged_on_call"] is True


def test_recovery_after_revive():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    s = next(x for x in results["scenarios"] if x["name"] == "recovery_after_revive")
    assert s["actual_mode"] == "model"
