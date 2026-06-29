import pytest
import pandas as pd
import numpy as np
import uuid
import os
from src.data_loader import load_and_validate_data, is_sensor_fault

def test_load_and_validate_fails_loudly_on_missing_columns(tmp_path):
    df = pd.DataFrame({'session_id': ['123'], 'student_id': ['456']})
    file_path = tmp_path / "bad_data.csv"
    df.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns in dataset"):
        load_and_validate_data(str(file_path))

def test_load_and_validate_fails_loudly_on_empty_data(tmp_path):
    df = pd.DataFrame()
    file_path = tmp_path / "empty_data.csv"
    df.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError, match="Data file is empty"):
        load_and_validate_data(str(file_path))

def test_deterministic_deduplication(tmp_path):
    from src.data_loader import REQUIRED_COLUMNS
    # Create valid df with duplicate session_id
    data = {col: [0, 0] for col in REQUIRED_COLUMNS}
    data['session_id'] = ['dup_123', 'dup_123']
    data['student_id'] = ['s1', 's2']
    
    df = pd.DataFrame(data)
    file_path = tmp_path / "dup_data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = load_and_validate_data(str(file_path))
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]['student_id'] == 's1' # keep first

def test_sensor_fault_identified_correctly():
    # All signals null or zero
    row1 = pd.Series({
        'tab_switch_count': 0,
        'face_count_anomalies': np.nan,
        'copy_paste_events': 0,
        'time_per_question_zscore': np.nan,
        'network_latency_flag': 0,
        'webcam_dropout_seconds': np.nan
    })
    
    assert is_sensor_fault(row1) == True
    
    # At least one signal is non-zero
    row2 = pd.Series({
        'tab_switch_count': 1,
        'face_count_anomalies': np.nan,
        'copy_paste_events': 0,
        'time_per_question_zscore': np.nan,
        'network_latency_flag': 0,
        'webcam_dropout_seconds': np.nan
    })
    assert is_sensor_fault(row2) == False
