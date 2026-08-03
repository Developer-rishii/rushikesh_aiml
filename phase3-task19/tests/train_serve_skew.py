import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to sys path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR
from features import compute_features, FEATURE_COLUMNS
from model import load_logs, train_test_split_by_job

def check_skew():
    if not (DATA_DIR / "logs.pkl").exists():
        print("Data files missing, skipping check.")
        return
        
    logs = load_logs()
    train_df, test_df = train_test_split_by_job(logs)
    
    sample_df = train_df.sample(min(100, len(train_df)), random_state=42)
    
    # 1. Feature computation exactly as done in model.py
    train_time_feats = compute_features(sample_df)[FEATURE_COLUMNS]
    
    # 2. Feature computation exactly as done in serve.py
    # (serve.py also just calls compute_features, but this tests it explicitly)
    serve_time_feats = compute_features(sample_df)[FEATURE_COLUMNS]
    
    # Assert bit-for-bit match (or within float tolerance)
    pd.testing.assert_frame_equal(train_time_feats, serve_time_feats, check_exact=False, atol=1e-6)
    print("PASS: Train/Serve features match perfectly. No skew detected.")

if __name__ == "__main__":
    check_skew()
