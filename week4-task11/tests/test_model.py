import pytest
import pandas as pd
import numpy as np
from src.model import train_and_evaluate, predict_session

def test_unreviewed_rows_excluded_from_evaluation(tmp_path):
    from src.data_loader import REQUIRED_COLUMNS
    
    # Create 5 rows: 1 unreviewed, 4 reviewed (2 of each class).
    data = {col: [0]*5 for col in REQUIRED_COLUMNS}
    data['ground_truth_reviewed'] = [np.nan, 1, 1, 1, 1]
    data['confirmed_violation'] = [np.nan, 1, 1, 0, 0]
    data['flagged_by_v0_proctor'] = [0, 1, 1, 0, 0]
    
    df = pd.DataFrame(data)
    
    # Run training
    model_dir = tmp_path / "models"
    log_entry, clf, imputer = train_and_evaluate(df, model_dir=str(model_dir))
    
    # The train_size + test_size should equal 4 (only the reviewed rows)
    assert log_entry['train_size'] + log_entry['test_size'] == 4


def test_borderline_confidence_is_lower(tmp_path):
    from src.data_loader import REQUIRED_COLUMNS
    
    # Let's create a dataset to train on
    data = {col: [0]*10 for col in REQUIRED_COLUMNS}
    df = pd.DataFrame(data)
    df['ground_truth_reviewed'] = 1
    
    # 5 strong violations, 5 clear non-violations
    df.loc[0:4, 'tab_switch_count'] = 10
    df.loc[0:4, 'face_count_anomalies'] = 5
    df.loc[0:4, 'confirmed_violation'] = 1
    
    df.loc[5:9, 'tab_switch_count'] = 0
    df.loc[5:9, 'face_count_anomalies'] = 0
    df.loc[5:9, 'confirmed_violation'] = 0
    
    df['flagged_by_v0_proctor'] = df['confirmed_violation']
    
    model_dir = tmp_path / "models"
    log_entry, clf, imputer = train_and_evaluate(df, model_dir=str(model_dir))
    
    # Create a strong violation row
    strong_row = pd.Series({col: 0 for col in REQUIRED_COLUMNS})
    strong_row['tab_switch_count'] = 10
    strong_row['face_count_anomalies'] = 5
    strong_row['flagged_by_v0_proctor'] = 1
    
    # Create a borderline row (one weak signal)
    borderline_row = pd.Series({col: 0 for col in REQUIRED_COLUMNS})
    borderline_row['tab_switch_count'] = 1
    borderline_row['face_count_anomalies'] = 0
    borderline_row['flagged_by_v0_proctor'] = 0
    
    res_strong = predict_session(strong_row, clf, imputer)
    res_borderline = predict_session(borderline_row, clf, imputer)
    
    # Confidence for strong violation should be higher than the borderline case (which should be closer to 0.5)
    # The borderline case might be predicted as 'clean' or 'flagged' but the confidence (distance from 0.5) should be lower.
    
    conf_strong = res_strong['model_score'] if res_strong['model_score'] > 0.5 else 1 - res_strong['model_score']
    conf_borderline = res_borderline['model_score'] if res_borderline['model_score'] > 0.5 else 1 - res_borderline['model_score']
    
    assert conf_strong > conf_borderline
