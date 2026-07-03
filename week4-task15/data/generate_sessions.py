import pandas as pd
import numpy as np
import uuid
import random
import os

def generate_data(num_records=400, output_path="d:/Placemux-aiml/week4-task15/data/flagged_sessions.csv"):
    np.random.seed(42)
    random.seed(42)

    student_ids = [f"stu_{i:04d}" for i in range(1, (num_records // 2) + 1)]
    assessment_ids = [f"asm_{i:03d}" for i in range(1, 10)]

    data = []
    
    num_clean = int(num_records * 0.4)
    for _ in range(num_clean):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(0, 3),
            "face_count_anomalies": 0,
            "copy_paste_events": np.random.randint(0, 2),
            "time_per_question_zscore": np.random.normal(0, 0.5),
            "network_latency_flag": np.random.choice([0, 1], p=[0.9, 0.1]),
            "webcam_dropout_seconds": np.random.randint(0, 5),
            "ground_truth_label": 0,
            "scenario": "normal_clean"
        })

    num_violations = int(num_records * 0.15)
    for _ in range(num_violations):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(5, 15),
            "face_count_anomalies": np.random.randint(2, 5),
            "copy_paste_events": np.random.randint(3, 10),
            "time_per_question_zscore": np.random.normal(-1.5, 0.5),
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0,
            "ground_truth_label": 1,
            "scenario": "true_violation"
        })

    num_fp_network = int(num_records * 0.1)
    for _ in range(num_fp_network):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(4, 8),
            "face_count_anomalies": 0,
            "copy_paste_events": 0,
            "time_per_question_zscore": np.random.normal(1.0, 0.5),
            "network_latency_flag": 1,
            "webcam_dropout_seconds": np.random.randint(10, 30),
            "ground_truth_label": 0,
            "scenario": "fp_network"
        })

    num_fp_cat = int(num_records * 0.05)
    for _ in range(num_fp_cat):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(0, 2),
            "face_count_anomalies": np.random.randint(1, 3),
            "copy_paste_events": 0,
            "time_per_question_zscore": np.random.normal(0, 0.5),
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0,
            "ground_truth_label": 0,
            "scenario": "fp_cat"
        })

    num_fp_cp = int(num_records * 0.1)
    for _ in range(num_fp_cp):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(0, 2),
            "face_count_anomalies": 0,
            "copy_paste_events": np.random.randint(3, 6),
            "time_per_question_zscore": np.random.normal(0.5, 0.5),
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0,
            "ground_truth_label": 0,
            "scenario": "fp_copypaste"
        })

    num_faults = int(num_records * 0.05)
    for _ in range(num_faults):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.nan,
            "face_count_anomalies": np.nan,
            "copy_paste_events": np.nan,
            "time_per_question_zscore": np.nan,
            "network_latency_flag": np.nan,
            "webcam_dropout_seconds": np.nan,
            "ground_truth_label": np.nan,
            "scenario": "sensor_fault"
        })

    num_borderline = int(num_records * 0.1)
    for _ in range(num_borderline):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": 4,
            "face_count_anomalies": 0,
            "copy_paste_events": 0,
            "time_per_question_zscore": np.random.normal(0, 0.5),
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0,
            "ground_truth_label": random.choice([0, 1]),
            "scenario": "borderline"
        })

    current_len = len(data)
    for _ in range(num_records - current_len):
        data.append({
            "session_id": str(uuid.uuid4()),
            "student_id": random.choice(student_ids),
            "assessment_id": random.choice(assessment_ids),
            "tab_switch_count": np.random.randint(0, 5),
            "face_count_anomalies": np.random.randint(0, 2),
            "copy_paste_events": np.random.randint(0, 4),
            "time_per_question_zscore": np.random.normal(0, 0.5),
            "network_latency_flag": 0,
            "webcam_dropout_seconds": 0,
            "ground_truth_label": np.nan,
            "scenario": "unlabeled"
        })

    df = pd.DataFrame(data)

    def compute_v0(row):
        if pd.isna(row["tab_switch_count"]) and pd.isna(row["face_count_anomalies"]) and pd.isna(row["copy_paste_events"]):
            return np.nan 
        if row["tab_switch_count"] > 3 or row["face_count_anomalies"] > 0 or row["copy_paste_events"] > 2:
            return 1
        return 0

    df["flagged_by_v0"] = df.apply(compute_v0, axis=1)

    # Need around 20-25% reviewed ground truth, rest null.
    # Currently almost 100% have ground truth. Let's sample indices to keep ground truth.
    num_labeled_target = int(num_records * 0.25)
    labeled_indices = df[df["ground_truth_label"].notna()].index.tolist()
    if len(labeled_indices) > num_labeled_target:
        # Prioritize keeping FP patterns and True violations
        critical_scenarios = ["true_violation", "fp_network", "fp_cat", "fp_copypaste"]
        critical_idx = df[df["scenario"].isin(critical_scenarios)].index.tolist()
        non_critical_idx = [i for i in labeled_indices if i not in critical_idx]
        
        # We need to drop (len(labeled_indices) - num_labeled_target)
        to_drop = len(labeled_indices) - num_labeled_target
        if to_drop <= len(non_critical_idx):
            drop_idx = random.sample(non_critical_idx, to_drop)
        else:
            drop_idx = non_critical_idx + random.sample(critical_idx, to_drop - len(non_critical_idx))
        
        df.loc[drop_idx, "ground_truth_label"] = np.nan

    import datetime
    base_time = datetime.datetime.now()
    df["timestamp"] = [base_time + datetime.timedelta(minutes=i) for i in range(len(df))]
    
    # 5 duplicates
    duplicates = df.sample(5).copy()
    duplicates["timestamp"] = [base_time + datetime.timedelta(minutes=len(df)+i) for i in range(len(duplicates))]
    
    df = pd.concat([df, duplicates], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic records at {output_path}")

if __name__ == "__main__":
    generate_data()
