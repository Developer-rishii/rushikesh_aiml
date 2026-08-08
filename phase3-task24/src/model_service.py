"""
Production-serving model wrapper. Exposes a kill switch and latency/error
injection so the chaos engine can simulate real outages instead of asserting
behaviour that was never actually exercised.
"""
import time
import joblib


class ModelServiceDownError(Exception):
    pass


class ModelService:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.is_down = False  # chaos hook

    def kill(self):
        self.is_down = True

    def revive(self):
        self.is_down = False

    def predict_score(self, feats: dict) -> float:
        if self.is_down:
            raise ModelServiceDownError("model service unavailable")
        x = [[feats["skill_match"], feats["exp_match"], feats["location_match"]]]
        return float(self.model.predict(x)[0])
