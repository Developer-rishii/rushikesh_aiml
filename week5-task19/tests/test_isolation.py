from fastapi.testclient import TestClient
import pandas as pd
import pytest

from src.api import app
import src.api

@pytest.fixture(autouse=True)
def mock_data():
    src.api.ITEMS_DF = pd.DataFrame({
        'item_id': ['ITM_1', 'ITM_2'],
        'subject': ['Math', 'Physics'],
        'allowed_colleges': ['COL_1,COL_2', 'COL_3']
    })
    
    src.api.FEATURES_SCORED_DF = pd.DataFrame({
        'item_id': ['ITM_1', 'ITM_2'],
        'response_count': [50, 50],
        'p_value': [0.5, 0.5],
        'point_biserial_corr': [0.5, 0.5],
        'bottom_25_correct_rate': [0.2, 0.2],
        'top_25_correct_rate': [0.8, 0.8],
        'subject_x': ['Math', 'Physics'],
        'model_is_weak': [False, True],
        'model_confidence': [0.1, 0.9]
    })
    
client = TestClient(app)

def test_isolation_blocked():
    # COL_1 tries to access ITM_2 which belongs to COL_3
    response = client.get("/college/COL_1/item/ITM_2")
    assert response.status_code == 403
    assert "does not have access" in response.json()['detail']

def test_isolation_allowed():
    # COL_1 tries to access ITM_1 which belongs to COL_1 and COL_2
    response = client.get("/college/COL_1/item/ITM_1")
    assert response.status_code == 200
    assert response.json()['item_id'] == 'ITM_1'
