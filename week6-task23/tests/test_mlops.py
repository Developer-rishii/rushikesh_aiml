import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_store import FeatureStore
from src.registry import ModelRegistry
from src.model import MatchModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def test_feature_store_online():
    fs = FeatureStore(DATA_DIR)
    # Use real ids from students.csv and jobs.csv if possible,
    # but for safety we can just test if the loading logic fails.
    # We will test an exception handling for invalid IDs.
    with pytest.raises(ValueError, match="not found in feature store"):
        fs.get_online_features("GHOST_STUDENT", "GHOST_JOB")

def test_model_registry_logging_and_promotion(tmp_path):
    registry = ModelRegistry()
    
    model1 = MatchModel("v99")
    model1.fit(pd.DataFrame([
        {"overlap_count": 1, "overlap_ratio": 1.0, "weighted_skill_score": 1.0, "years_gap": 0, "missing_top_skill": 0, "jd_breadth": 1, "student_breadth": 1, "good_match": 1},
        {"overlap_count": 0, "overlap_ratio": 0.0, "weighted_skill_score": 0.0, "years_gap": -2, "missing_top_skill": 1, "jd_breadth": 1, "student_breadth": 1, "good_match": 0}
    ]))
    
    registry.log_model(model1, "v99", {"precision": 0.99})
    
    metadata = registry._load_metadata()
    assert "v99" in metadata["models"]
    assert metadata["models"]["v99"]["metrics"]["precision"] == 0.99
    
    registry.promote_to_production("v99")
    
    metadata_after = registry._load_metadata()
    assert metadata_after["production_version"] == "v99"
    assert metadata_after["models"]["v99"]["status"] == "Production"
