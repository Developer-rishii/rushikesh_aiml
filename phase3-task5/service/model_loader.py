import lightgbm as lgb
import os

_MODEL_INSTANCE = None

def load_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        model_path = 'models/lgbm_ranker.txt'
        if os.path.exists(model_path):
            _MODEL_INSTANCE = lgb.Booster(model_file=model_path)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    return _MODEL_INSTANCE
