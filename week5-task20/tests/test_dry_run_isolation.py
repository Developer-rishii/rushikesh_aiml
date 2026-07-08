import pytest
import os
import json

def test_dry_run_transcript_validates_isolation():
    """
    Proves cross-college rejection actually happens during the scripted dry run, 
    not just in a unit test in isolation.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcript_path = os.path.join(base, "reports", "dry_run_transcript.json")
    assert os.path.exists(transcript_path), "Dry run transcript missing"
    
    with open(transcript_path) as f:
        data = json.load(f)
        
    transcript = data["transcript"]
    
    # Find isolation steps
    isolation_steps = [t for t in transcript if "Cross-college" in t["description"]]
    
    assert len(isolation_steps) >= 3, "Not enough cross-college isolation checks in dry run"
    
    for step in isolation_steps:
        # Step must have passed
        assert step["passed"] is True
        # Actual status must be 403 Forbidden
        assert step["actual_status"] == 403
        # Response body must contain access denied message
        assert "Access denied" in str(step["response_body"])
