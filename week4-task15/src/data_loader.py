import pandas as pd

REQUIRED_COLUMNS = [
    "session_id", "student_id", "assessment_id",
    "tab_switch_count", "face_count_anomalies", "copy_paste_events",
    "time_per_question_zscore", "network_latency_flag", "webcam_dropout_seconds",
    "flagged_by_v0", "timestamp"
]

def load_and_validate_data(filepath="d:/Placemux-aiml/week4-task15/data/flagged_sessions.csv"):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Flagged session data missing at {filepath}. Upstream pipeline failed.")
    
    if df.empty:
        raise ValueError("Flagged session data is completely empty. Upstream pipeline produced no records.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Upstream flagged-session data is malformed. Missing columns: {missing_cols}")
    
    # Check for all-null batches (excluding sensor faults which might be intentional rows, but if ALL are null, something is wrong)
    signal_cols = ["tab_switch_count", "face_count_anomalies", "copy_paste_events"]
    if df[signal_cols].isna().all().all():
        raise ValueError("Upstream flagged-session data consists entirely of null signals (empty batches).")

    # Deduplicate based on session_id, keeping the last timestamp
    initial_len = len(df)
    # Sort by timestamp to ensure 'last' is actually the latest
    df = df.sort_values(by="timestamp")
    df = df.drop_duplicates(subset=["session_id"], keep="last")
    deduped_len = len(df)
    
    print(f"Loaded {deduped_len} valid sessions. Deduplicated {initial_len - deduped_len} rows.")
    return df

if __name__ == "__main__":
    df = load_and_validate_data()
    print(df.head())
