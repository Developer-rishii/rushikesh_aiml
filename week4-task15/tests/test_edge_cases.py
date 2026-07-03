import pytest
import pandas as pd
import numpy as np
import uuid
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_and_validate_data, REQUIRED_COLUMNS
from src.explainer import explain_prediction
import joblib

# Paths
TEST_DATA_DIR = "d:/Placemux-aiml/week4-task15/tests/test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def test_upstream_dependency_validation():
    # Create malformed data (missing columns)
    malformed_path = os.path.join(TEST_DATA_DIR, "malformed.csv")
    df = pd.DataFrame({"session_id": ["123"], "student_id": ["stu_1"]})
    df.to_csv(malformed_path, index=False)
    
    with pytest.raises(ValueError, match="(?i)missing columns") as excinfo:
        load_and_validate_data(malformed_path)
    
    assert "malformed" in str(excinfo.value).lower()
    
    # Test all null batches
    all_null_path = os.path.join(TEST_DATA_DIR, "all_null.csv")
    df_null = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df_null.loc[0] = [np.nan] * len(REQUIRED_COLUMNS)
    df_null["session_id"] = "123"
    df_null["timestamp"] = pd.Timestamp.now()
    df_null.to_csv(all_null_path, index=False)
    
    with pytest.raises(ValueError, match="entirely of null signals"):
        load_and_validate_data(all_null_path)

def test_sensor_fault_routing():
    fault_session = {
        "tab_switch_count": np.nan,
        "face_count_anomalies": np.nan,
        "copy_paste_events": np.nan,
        "time_per_question_zscore": np.nan,
        "network_latency_flag": np.nan,
        "webcam_dropout_seconds": np.nan
    }
    result = explain_prediction(fault_session)
    assert result["verdict"] == "no_data"
    assert "Sensor fault" in result["reason"]
    assert result["prediction"] is None

def test_duplicate_session_handling():
    # Duplicate session ID should be deduplicated by keep="last"
    dupe_path = os.path.join(TEST_DATA_DIR, "dupes.csv")
    df = pd.DataFrame([
        {"session_id": "sid_1", "timestamp": "2023-01-01T10:00:00", "tab_switch_count": 1},
        {"session_id": "sid_1", "timestamp": "2023-01-01T11:00:00", "tab_switch_count": 2}, # This should be kept
    ])
    # Add other required cols to not fail validation
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            
    df.to_csv(dupe_path, index=False)
    df_loaded = load_and_validate_data(dupe_path)
    assert len(df_loaded) == 1
    assert df_loaded.iloc[0]["tab_switch_count"] == 2

def test_threshold_edge_case():
    model_path = "d:/Placemux-aiml/week4-task15/src/models/proctor_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")
        
    model_data = joblib.load(model_path)
    model = model_data["model"]
    features = model_data["features"]
    threshold = model_data["threshold"]
    
    # We test that a prediction exactly at threshold or slightly above/below is deterministic
    # We will simulate the `explain_prediction` logic
    test_session = {f: 0 for f in features}
    df = pd.DataFrame([test_session])
    X = df[features]
    prob = model.predict_proba(X)[0][1]
    
    # Since we can't easily force probability to exactly equal threshold,
    # we just verify the logic treats threshold as inclusive (>= threshold is flagged).
    # We can mock the predict_proba.
    from unittest.mock import patch
    
    class MockModel:
        def predict_proba(self, X):
            return np.array([[1 - threshold, threshold]])
            
    # Mocking for test
    model_data_mocked = {"model": MockModel(), "features": features, "threshold": threshold}
    
    with patch("src.explainer.joblib.load", return_value=model_data_mocked):
        test_input = {f: 0 for f in features}
        test_input.update({
            "tab_switch_count": 0,
            "face_count_anomalies": 0,
            "copy_paste_events": 0,
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0
        })
        res = explain_prediction(test_input, "d:/Placemux-aiml/week4-task15/src/models/mock_model.pkl")
        assert res["verdict"] == "flagged"

def test_fp_proof_of_detection():
    # Known FP: network issue
    fp_session = {
        "tab_switch_count": 7, # high tab switch
        "face_count_anomalies": 0,
        "copy_paste_events": 0,
        "time_per_question_zscore": 1.0,
        "network_latency_flag": 1,
        "webcam_dropout_seconds": 15
    }
    # In v0, this would flag because tab_switch_count > 3
    # Our model should clear it
    res = explain_prediction(fp_session)
    assert res["verdict"] == "cleared"
    assert res["fp_pattern"] == "network_issue"
    assert res["prediction"] == 0

def test_tp_proof_of_detection():
    # Known TP: True violation
    tp_session = {
        "tab_switch_count": 10,
        "face_count_anomalies": 3,
        "copy_paste_events": 5,
        "time_per_question_zscore": -1.5,
        "network_latency_flag": 0,
        "webcam_dropout_seconds": 0
    }
    res = explain_prediction(tp_session)
    assert res["verdict"] == "flagged"
    assert res["prediction"] == 1
