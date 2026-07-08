import pandas as pd
import numpy as np
from src.features import extract_features

def test_schema_validation():
    # Test that invalid rows don't break feature extraction and are handled loudly/safely
    items_df = pd.DataFrame({
        'item_id': ['ITM_1', 'ITM_2'],
        'subject': ['Math', 'Physics'],
        'true_a': [1.0, 1.0],
        'true_b': [0.0, 0.0],
        'is_weak_item': [False, False],
        'split': ['train', 'train'],
        'allowed_colleges': ['COL_1', 'COL_1']
    })
    
    responses_df = pd.DataFrame({
        'response_id': ['R_1', 'R_2', 'R_3', 'R_4'],
        'student_id': ['S_1', 'S_2', np.nan, 'S_3'], # nan student
        'item_id': ['ITM_1', 'ITM_1', 'ITM_1', 'ITM_2'],
        'college_id': ['COL_1', 'COL_1', 'COL_1', 'COL_1'],
        'correct': [1, 'invalid_string', 0, 1], # invalid correct type
        'time_spent_sec': [45, 45, 45, -10] # invalid time
    })
    
    # Feature extraction should silently drop or loudly fail malformed rows
    # In our implementation, `clean_data` drops them safely so the pipeline continues
    features = extract_features(items_df, responses_df)
    
    # ITM_1 should only have 1 valid response (R_1)
    itm1_feats = features[features['item_id'] == 'ITM_1'].iloc[0]
    assert itm1_feats['response_count'] == 1
    
    # ITM_2 should have 0 valid responses because time_spent was -10
    itm2_feats = features[features['item_id'] == 'ITM_2'].iloc[0]
    assert itm2_feats['response_count'] == 0
