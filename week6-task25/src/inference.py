"""
inference.py

Serving logic shared by the FastAPI layer (src/api/main.py) and the
monitoring simulation (scripts/simulate_and_monitor.py), so "the model
running live" and "the model we demo" are provably the same code path -
not a second copy that only looks right in a demo.

Also implements Section 4's "Explainability" requirement: for every
prediction we return the top feature contributions and a plain-English
sentence, computed directly from the logistic regression's own
coefficients (scaled-feature * coefficient), which is an exact
decomposition of the model's own logit - not a post-hoc approximation.
"""

import json
import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH

FEATURE_LABELS = {
    "skill_overlap_score": "verified skill overlap with the JD",
    "years_experience": "years of relevant experience",
    "experience_gap": "gap between required and actual experience",
    "resume_parse_confidence": "resume/JD parsing confidence",
    "interview_eval_score": "interview evaluation score",
    "communication_score": "verified communication score",
    "role_historical_hire_rate": "historical hire rate for this role segment",
}


class MatchModel:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["decision_threshold"]
        self.feature_columns = bundle["feature_columns"]
        self.scaler = self.pipeline.named_steps["scaler"]
        self.clf = self.pipeline.named_steps["clf"]

    def validate(self, record: dict):
        """Raise a clear error for malformed input instead of letting sklearn
        throw an opaque exception deep in the pipeline (Section 8 -
        'errors handled')."""
        missing = [c for c in self.feature_columns if c not in record]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        bad_type = [c for c in self.feature_columns
                    if not isinstance(record[c], (int, float, np.integer, np.floating))
                    or isinstance(record[c], bool)]
        if bad_type:
            raise ValueError(f"Non-numeric value for feature(s): {bad_type}")
        non_finite = [c for c in self.feature_columns if not np.isfinite(record[c])]
        if non_finite:
            raise ValueError(f"NaN/Inf value for feature(s): {non_finite}")

    def predict_one(self, record: dict, explain=True):
        self.validate(record)
        x = pd.DataFrame([record])[self.feature_columns]
        proba = float(self.pipeline.predict_proba(x)[0, 1])
        label = int(proba >= self.threshold)
        result = {
            "match_probability": round(proba, 4),
            "predicted_label": label,
            "decision_threshold": self.threshold,
        }
        if explain:
            result["explanation"] = self.explain(record, proba, label)
        return result

    def predict_batch(self, df: pd.DataFrame):
        x = df[self.feature_columns]
        proba = self.pipeline.predict_proba(x)[:, 1]
        label = (proba >= self.threshold).astype(int)
        return proba, label

    def explain(self, record: dict, proba: float, label: int):
        """Exact per-feature contribution to the logit, from the model's own
        scaler + coefficients, plus a plain-English sentence."""
        x = pd.DataFrame([record])[self.feature_columns]
        x_scaled = self.scaler.transform(x)[0]
        coefs = self.clf.coef_[0]
        contributions = x_scaled * coefs
        order = np.argsort(-np.abs(contributions))

        top = []
        for i in order[:3]:
            feat = self.feature_columns[i]
            top.append({
                "feature": feat,
                "value": round(float(record[feat]), 3),
                "contribution": round(float(contributions[i]), 4),
                "direction": "supports match" if contributions[i] > 0 else "against match",
            })

        verdict = "a likely successful match" if label == 1 else "an unlikely match"
        lead_feat = FEATURE_LABELS.get(top[0]["feature"], top[0]["feature"])
        lead_dir = "strong" if top[0]["direction"] == "supports match" else "weak"
        second_feat = FEATURE_LABELS.get(top[1]["feature"], top[1]["feature"])
        sentence = (
            f"Scored {proba:.0%} probability -> classified as {verdict}. "
            f"Main driver: {lead_dir} {lead_feat} ({record[top[0]['feature']]:.2f}); "
            f"secondary factor: {second_feat} ({record[top[1]['feature']]:.2f}, "
            f"{top[1]['direction']})."
        )
        return {"summary": sentence, "top_features": top}


def load_default_model():
    return MatchModel(MODEL_PATH)
