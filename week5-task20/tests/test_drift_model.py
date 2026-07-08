import pytest
import os
import json

def test_drift_auc_sane():
    """
    Regression guard: drift classifier's train/test split doesn't leak, 
    and AUC is not a suspicious constant (e.g., exactly 1.0 or exactly 0.5 every run).
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base, "reports", "drift_results.json")
    assert os.path.exists(report_path), "Drift report missing"
    
    with open(report_path) as f:
        data = json.load(f)
        
    auc = data["drift_auc"]
    assert isinstance(auc, float)
    
    # Must be > 0.0 and <= 1.0
    assert 0.0 <= auc <= 1.0
    
    # Not suspiciously perfect 1.0
    assert auc < 0.999, "Suspiciously perfect drift AUC (1.0). Target leakage likely."
    
    # Not suspiciously perfect 0.5000 
    assert auc != 0.5000, "Suspiciously perfect 0.5 AUC. Model didn't learn anything."

def test_ks_tests_run():
    """Ensure per-feature KS-tests were actually computed."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base, "reports", "drift_results.json")
    
    with open(report_path) as f:
        data = json.load(f)
        
    ks = data["ks_test_results"]
    assert "match_score" in ks
    assert "p_value" in ks["match_score"]
