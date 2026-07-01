import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_data(num_samples=2000, output_path="synthetic_sessions.csv"):
    """
    Generates a realistic synthetic dataset for proctoring sessions.
    Label: 'true_violation' (30%) vs 'false_positive' (70%).
    """
    np.random.seed(42)
    random.seed(42)
    
    # Target distribution
    is_violation = np.random.choice([0, 1], size=num_samples, p=[0.70, 0.30])
    
    # Generate features
    data = []
    for i in range(num_samples):
        violation = is_violation[i]
        
        # Features with different distributions based on class
        if violation:
            # Real violations have higher metrics generally
            tab_switch_count = np.random.poisson(lam=5)
            face_absent_seconds = np.random.exponential(scale=15)
            multiple_faces_detected = np.random.choice([0, 1], p=[0.6, 0.4])
            audio_anomaly_score = np.random.uniform(0.5, 1.0)
            eye_gaze_offscreen_pct = np.random.normal(loc=40, scale=15)
        else:
            # False positives are often triggered by smaller blips
            tab_switch_count = np.random.poisson(lam=1)
            face_absent_seconds = np.random.exponential(scale=3)
            multiple_faces_detected = np.random.choice([0, 1], p=[0.95, 0.05])
            audio_anomaly_score = np.random.uniform(0.0, 0.6)
            eye_gaze_offscreen_pct = np.random.normal(loc=15, scale=10)
            
        # Common features
        device_type = np.random.choice(['desktop', 'laptop', 'mobile'])
        network_quality = np.random.choice(['excellent', 'good', 'poor'], p=[0.4, 0.4, 0.2])
        session_duration = max(600, np.random.normal(loc=3600, scale=600)) # in seconds
        time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'])
        candidate_history_flag_rate = np.random.uniform(0.0, 0.3)
        
        # Clip values
        eye_gaze_offscreen_pct = max(0, min(100, eye_gaze_offscreen_pct))
        
        # Add some missing/noisy data to force edge case handling
        if random.random() < 0.05:
            face_absent_seconds = np.nan
        if random.random() < 0.03:
            audio_anomaly_score = np.nan
            
        row = {
            'session_id': f"sess_{i:04d}",
            'tab_switch_count': tab_switch_count,
            'face_absent_seconds': face_absent_seconds,
            'multiple_faces_detected': multiple_faces_detected,
            'audio_anomaly_score': audio_anomaly_score,
            'eye_gaze_offscreen_pct': eye_gaze_offscreen_pct,
            'device_type': device_type,
            'network_quality': network_quality,
            'session_duration': session_duration,
            'time_of_day': time_of_day,
            'candidate_history_flag_rate': candidate_history_flag_rate,
            'label': 'true_violation' if violation else 'false_positive'
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Introduce one completely malformed row
    df.loc[num_samples - 1, 'session_id'] = "sess_malformed_123"
    df.loc[num_samples - 1, 'tab_switch_count'] = -999 # adversarial value
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} synthetic sessions and saved to {output_path}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")

def load_data(filepath="synthetic_sessions.csv"):
    """
    Loader function. Can be easily swapped out for real CSV data once available,
    provided the schema matches.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Please generate synthetic data first or provide the real data drop-in.")
    
    return pd.read_csv(filepath)

if __name__ == "__main__":
    generate_synthetic_data(num_samples=2000, output_path="d:/Placemux-aiml/week4-task13/data/synthetic_sessions.csv")
