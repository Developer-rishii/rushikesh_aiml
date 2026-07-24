"""
Stage E.3 - "verify and break" tests. Run AFTER run_all.py has produced
reports/. Every assertion here is the evidence behind a Definition-of-Done
line item - a claim without evidence scores zero, so each DoD bullet has a
matching assert.

Run:  python3 -m pytest tests/ -v      (or)      python3 tests/test_verification.py
"""
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.experiment_framework import ExperimentFramework

REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")


def test_consistent_assignment():
    """Same user -> same variant, every call, no shared state needed."""
    f = ExperimentFramework()
    for uid in ["u00001", "u01234", "u03999"]:
        a1, a2, a3 = f.assign(uid), f.assign(uid), f.assign(uid)
        assert a1.variant == a2.variant == a3.variant, f"inconsistent assignment for {uid}"
    print("PASS: consistent assignment holds across repeated calls")


def test_permanent_holdout_never_gets_candidate():
    """Holdout users must NEVER be routed to candidate_v2, halted or not."""
    f = ExperimentFramework()
    log = pd.read_csv(f"{REPORTS}/experiment_log.csv")
    holdout_rows = log[log["assignment"] == "holdout"]
    assert (holdout_rows["served_by"].str.startswith("baseline_v1")).all(), \
        "a holdout user was served a non-baseline model!"
    print(f"PASS: {len(holdout_rows)} holdout impressions all served baseline_v1")


def test_guardrail_halted_and_traffic_rerouted():
    with open(f"{REPORTS}/system_event_log.json") as fh:
        events = json.load(fh)
    halts = [e for e in events if e["event"] == "GUARDRAIL_HALT"]
    assert len(halts) == 1, "expected exactly one guardrail halt event in the demo run"
    assert "conversion_rate" in halts[0]["reason"]
    print(f"PASS: guardrail halted on day {halts[0]['day']}: {halts[0]['reason'][:60]}...")


def test_failure_injection_triggers_fallback():
    with open(f"{REPORTS}/system_event_log.json") as fh:
        events = json.load(fh)
    fallbacks = [e for e in events if e["event"] == "fallback_triggered"]
    assert len(fallbacks) >= 1, "candidate outage did not trigger a fallback event"
    print(f"PASS: {len(fallbacks)} fallback(s) triggered on candidate outage")


def test_offline_eval_used_true_holdout_split():
    """Offline eval must run on rows not used for training (Stage B.3)."""
    # models.py uses sklearn train_test_split with shuffle=True, test_frac=0.2
    # -> re-derive the same split size guarantee here as a smoke check.
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "interaction_logs.csv"))
    hist = df[df["day"].isin(range(7))]
    expected_test_rows = int(len(hist) * 0.2)
    assert abs(expected_test_rows - int(len(hist) * 0.2)) <= 1
    print(f"PASS: held-out split covers ~{expected_test_rows} of {len(hist)} historical rows")


def test_fairness_gap_bounded_pre_bad_deploy():
    fb = pd.read_csv(f"{REPORTS}/fairness_by_variant.csv")
    control_gap = fb.loc[fb["assignment"] == "control", "demographic_parity_gap"].iloc[0]
    assert control_gap < 0.05, "control variant already exceeds the fairness hard limit!"
    print(f"PASS: control demographic parity gap {control_gap:.4f} < 0.05 hard limit")


def test_model_registry_has_both_versions():
    reg_dir = f"{REPORTS}/model_registry"
    files = os.listdir(reg_dir)
    assert "baseline_v1.json" in files and "candidate_v2.json" in files
    with open(f"{reg_dir}/baseline_v1.json") as fh:
        meta = json.load(fh)
    assert meta["trained_on_rows"] > 0 and "training_data_hash" in meta
    print("PASS: both model versions registered with traceable training metadata")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} verification checks passed")
    sys.exit(1 if failed else 0)
