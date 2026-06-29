import pandas as pd

REQUIRED_COLUMNS = [
    'session_id', 'student_id', 'assessment_id', 
    'tab_switch_count', 'face_count_anomalies', 
    'copy_paste_events', 'time_per_question_zscore', 
    'network_latency_flag', 'webcam_dropout_seconds', 
    'flagged_by_v0_proctor', 'ground_truth_reviewed', 'confirmed_violation'
]

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads integrity data from a CSV, validates its structure, and handles deduplication.
    Fails loudly if required columns are missing or the file is empty.
    """
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to load data from {filepath}: {e}")

    if df.empty:
        raise ValueError(f"Data file is empty: {filepath}")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Deterministic deduplication by session_id
    initial_count = len(df)
    df = df.drop_duplicates(subset=['session_id'], keep='first')
    df = df.reset_index(drop=True)
    
    return df

def is_sensor_fault(row: pd.Series) -> bool:
    """
    Identifies a sensor fault: session where all relevant signal values are null or zero.
    """
    signals = ['tab_switch_count', 'face_count_anomalies', 'copy_paste_events', 'time_per_question_zscore', 'network_latency_flag', 'webcam_dropout_seconds']
    
    for sig in signals:
        val = row.get(sig)
        if pd.notna(val) and val != 0:
            return False
    return True
