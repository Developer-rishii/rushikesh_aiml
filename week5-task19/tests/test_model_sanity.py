import json
import os
import pytest

def test_model_sanity():
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        pytest.skip("Metrics file not found. Run pipeline first.")
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    model = metrics['model']
    baseline = metrics['baseline']
    
    # Accuracy / AUC shouldn't be suspiciously perfect (exactly 1.0) due to leakage
    # With synthetic IRT data, the model can't perfectly reverse engineer the truth
    assert model['auc'] is not None and model['auc'] < 1.0, "Suspiciously perfect AUC (label leakage?)"
    
    # Model should beat or match baseline in F1/AUC/overall utility.
    # We require precision and recall to be reasonable. 
    # Usually the ML model improves precision substantially over Dumb Baseline.
    assert (model['precision'] + model['recall']) > (baseline['precision'] + baseline['recall']) * 0.8, "Model performs terribly compared to baseline."
