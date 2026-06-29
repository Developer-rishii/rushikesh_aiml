import pandas as pd
import numpy as np
import uuid
import random

def generate_synthetic_data(num_records=350, output_path="integrity_data_week1.csv"):
    np.random.seed(42)
    random.seed(42)

    # Base features
    session_ids = [str(uuid.uuid4()) for _ in range(num_records)]
    student_ids = [str(uuid.uuid4()) for _ in range(num_records)]
    assessment_ids = np.random.choice(['MATH_101', 'CS_202', 'LIT_301'], size=num_records)

    tab_switch_count = np.random.poisson(lam=1.5, size=num_records)
    face_count_anomalies = np.random.poisson(lam=0.2, size=num_records)
    copy_paste_events = np.random.poisson(lam=0.5, size=num_records)
    time_per_question_zscore = np.random.normal(0, 1, size=num_records)
    network_latency_flag = np.random.choice([0, 1], p=[0.9, 0.1], size=num_records)
    webcam_dropout_seconds = np.random.exponential(scale=2.0, size=num_records)

    df = pd.DataFrame({
        'session_id': session_ids,
        'student_id': student_ids,
        'assessment_id': assessment_ids,
        'tab_switch_count': tab_switch_count,
        'face_count_anomalies': face_count_anomalies,
        'copy_paste_events': copy_paste_events,
        'time_per_question_zscore': time_per_question_zscore,
        'network_latency_flag': network_latency_flag,
        'webcam_dropout_seconds': webcam_dropout_seconds
    })

    # Inject edge cases
    
    # 1. Missing webcam data (camera permission denied / missing)
    missing_cam_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_cam_indices, 'webcam_dropout_seconds'] = np.nan
    df.loc[missing_cam_indices, 'face_count_anomalies'] = np.nan

    # 2. Sensor fault (all values null/zero - not a real signal)
    sensor_fault_index = missing_cam_indices[0] # Pick one to be complete fault
    df.loc[sensor_fault_index, 'tab_switch_count'] = np.nan
    df.loc[sensor_fault_index, 'face_count_anomalies'] = np.nan
    df.loc[sensor_fault_index, 'copy_paste_events'] = np.nan
    df.loc[sensor_fault_index, 'time_per_question_zscore'] = np.nan
    df.loc[sensor_fault_index, 'network_latency_flag'] = np.nan
    df.loc[sensor_fault_index, 'webcam_dropout_seconds'] = np.nan
    
    # Alternatively, 0 instead of NaN for count features
    sensor_fault_index2 = missing_cam_indices[1]
    df.loc[sensor_fault_index2, ['tab_switch_count', 'face_count_anomalies', 'copy_paste_events', 'network_latency_flag', 'webcam_dropout_seconds']] = 0
    df.loc[sensor_fault_index2, 'time_per_question_zscore'] = np.nan

    # 3. Duplicate session_ids
    # Duplicate 3 rows
    dup_indices = np.random.choice(df.index, size=3, replace=False)
    df_dups = df.loc[dup_indices].copy()
    df = pd.concat([df, df_dups], ignore_index=True)
    
    # 4. Borderline rows (one weak signal only)
    borderline_indices = np.random.choice(df.index, size=10, replace=False)
    for idx in borderline_indices:
        df.loc[idx, 'tab_switch_count'] = 1
        df.loc[idx, 'face_count_anomalies'] = 0
        df.loc[idx, 'copy_paste_events'] = 0
        df.loc[idx, 'time_per_question_zscore'] = 1.1

    # v0 Baseline Rule
    # Flag if tab_switch_count > 3 OR face_count_anomalies > 0
    df['flagged_by_v0_proctor'] = ((df['tab_switch_count'] > 3) | (df['face_count_anomalies'] > 0)).astype(int)

    # Ground truth labels
    # ~20% manually reviewed
    reviewed_indices = np.random.choice(df.index, size=int(len(df) * 0.20), replace=False)
    
    df['ground_truth_reviewed'] = np.nan
    df.loc[reviewed_indices, 'ground_truth_reviewed'] = 1
    
    df['confirmed_violation'] = np.nan
    
    # Assign logic to true violations based on features to allow a model to learn it
    # We want a model that considers more features than v0. 
    # e.g. violation if (tab_switches > 2 AND time_zscore > 1.5) OR (copy_paste > 0)
    for idx in reviewed_indices:
        is_violation = False
        row = df.loc[idx]
        if (pd.notna(row['tab_switch_count']) and row['tab_switch_count'] > 2 and pd.notna(row['time_per_question_zscore']) and row['time_per_question_zscore'] > 1.5):
            is_violation = True
        elif pd.notna(row['copy_paste_events']) and row['copy_paste_events'] > 0:
            is_violation = True
        elif pd.notna(row['face_count_anomalies']) and row['face_count_anomalies'] > 1:
             is_violation = True
             
        # Add some noise
        if random.random() < 0.1:
            is_violation = not is_violation
            
        df.loc[idx, 'confirmed_violation'] = int(is_violation)

    # Specifically ensure the sensor fault row is NOT in the reviewed set (or if it is, let it be not a violation or handled differently)
    # Let's keep it unreviewed for simplicity, or reviewed=1 violation=0.
    
    # Re-shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data(output_path="d:/Placemux-aiml/wek4-task11/data/integrity_data_week1.csv")
