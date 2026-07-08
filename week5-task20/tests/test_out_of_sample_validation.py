import pytest
import os
import pandas as pd
import hashlib

def test_out_of_sample_data_is_fresh():
    """
    Ensure the validation pipeline runs on genuinely different data 
    than the original training set (assert the fresh dataset's row-level 
    hash/seed differs from the original).
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fresh_path = os.path.join(base, "data", "fresh_matching.csv")
    orig_path = os.path.join(base, "..", "week5-task16", "data", "matching_v1_output.csv")
    
    assert os.path.exists(fresh_path), "Fresh data not found. Run pipeline first."
    assert os.path.exists(orig_path), "Original data not found."
    
    fresh_df = pd.read_csv(fresh_path)
    orig_df = pd.read_csv(orig_path)
    
    # 1. Size should differ (scale up)
    assert len(fresh_df) > len(orig_df), "Fresh dataset must be larger than original"
    
    # 2. Hash must differ
    fresh_hash = hashlib.sha256(fresh_df.to_csv(index=False).encode()).hexdigest()
    orig_hash = hashlib.sha256(orig_df.to_csv(index=False).encode()).hexdigest()
    assert fresh_hash != orig_hash, "Fresh dataset is identical to original training data!"
    
    # 3. New colleges must exist
    fresh_colleges = set(fresh_df["college_id"].unique())
    orig_colleges = set(orig_df["college_id"].unique())
    assert len(fresh_colleges - orig_colleges) > 0, "No new colleges generated in fresh data"

def test_metrics_are_sane():
    """Ensure OOS metrics are non-degenerate."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import json
    metrics_path = os.path.join(base, "reports", "metrics.json")
    assert os.path.exists(metrics_path), "Metrics report missing"
    
    with open(metrics_path) as f:
        metrics = json.load(f)
        
    m = metrics["fresh_data_model"]
    assert "precision_at_5" in m
    assert 0.0 <= m["precision_at_5"] <= 1.0
    assert 0.0 <= m["recall_at_5"] <= 1.0
    assert 0.0 <= m["fpr_at_5"] <= 1.0
    
    # Assert segments exist
    segments = metrics["segments"]
    assert "college_A" in segments
    assert any(k.startswith("seniority_") for k in segments.keys())
    assert any(k.startswith("college_size_") for k in segments.keys())
