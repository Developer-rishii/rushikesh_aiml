import pandas as pd
from src.features import extract_features
from src.explainability import generate_explanation

def test_zero_response_item():
    items = pd.DataFrame({'item_id': ['ITM_0'], 'subject': ['Math'], 'is_weak_item': [False], 'split': ['train']})
    responses = pd.DataFrame(columns=['response_id', 'student_id', 'item_id', 'college_id', 'correct', 'time_spent_sec'])
    features = extract_features(items, responses)
    assert features.iloc[0]['response_count'] == 0
    
    exp = generate_explanation(features.iloc[0].to_dict(), False)
    assert exp['status'] == 'needs_more_data'

def test_single_outcome_item():
    items = pd.DataFrame({'item_id': ['ITM_1'], 'subject': ['Math'], 'is_weak_item': [False], 'split': ['train']})
    responses = pd.DataFrame({
        'response_id': ['R1', 'R2', 'R3', 'R4'],
        'student_id': ['S1', 'S2', 'S3', 'S4'],
        'item_id': ['ITM_1', 'ITM_1', 'ITM_1', 'ITM_1'],
        'college_id': ['C1', 'C1', 'C1', 'C1'],
        'correct': [1, 1, 1, 1],
        'time_spent_sec': [45, 50, 40, 55]
    })
    
    # Should not crash on point biserial
    features = extract_features(items, responses)
    assert features.iloc[0]['p_value'] == 1.0
    assert pd.notna(features.iloc[0]['point_biserial_corr'])
    
def test_cold_start():
    # 5 responses
    items = pd.DataFrame({'item_id': ['ITM_2'], 'subject': ['Math'], 'is_weak_item': [False], 'split': ['train']})
    responses = pd.DataFrame({
        'response_id': ['R1', 'R2', 'R3', 'R4', 'R5'],
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'item_id': ['ITM_2']*5,
        'college_id': ['C1']*5,
        'correct': [1, 0, 1, 0, 1],
        'time_spent_sec': [45, 50, 40, 55, 45]
    })
    features = extract_features(items, responses)
    exp = generate_explanation(features.iloc[0].to_dict(), False)
    assert exp['status'] == 'needs_more_data'
    assert 'Need at least 20' in exp['admin_view']
