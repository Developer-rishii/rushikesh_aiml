"""
predict_pass_probability.py
===========================
Clean prediction interface for the Assessment Difficulty Prediction model.
Loads the saved model + feature engineer and provides single / batch
predictions with confidence levels, risk factors, and recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FeatureEngineer


class PassPredictor:
    """
    End-user prediction class.

    Usage
    -----
    >>> predictor = PassPredictor()
    >>> predictor.load_model()
    >>> result = predictor.predict_single({...})
    """

    def __init__(self,
                 model_path: str = None,
                 fe_path: str = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base, "models",
                                                     "best_model.pkl")
        self.fe_path = fe_path or os.path.join(base, "models",
                                               "feature_engineer.pkl")
        self.model = None
        self.feature_engineer: FeatureEngineer | None = None

    # Loading

    def load_model(self) -> None:
        """Load the trained model and fitted FeatureEngineer from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run model_training.py first."
            )
        self.model = joblib.load(self.model_path)
        self.feature_engineer = joblib.load(self.fe_path)
        print(f"[PassPredictor] Model loaded from {self.model_path}")
        print(f"[PassPredictor] FeatureEngineer loaded from {self.fe_path}")

    # Single prediction

    def predict_single(self, student_data: dict) -> dict:
        """
        Predict pass probability for a single student.

        Parameters
        ----------
        student_data : dict
            Must contain the raw columns expected by FeatureEngineer
            (subject can be an int already or will be set to 0 by default).

        Returns
        -------
        dict with keys:
            pass_probability (0-100), will_pass (bool),
            confidence ("High"/"Medium"/"Low"),
            risk_factors (list[str]), recommendations (list[str])
        """
        self._ensure_loaded()

        # Build single-row DataFrame
        df = pd.DataFrame([student_data])
        features = self.feature_engineer.transform(df)

        prob = self.model.predict_proba(features)[0, 1]
        pass_prob = round(prob * 100, 2)
        will_pass = pass_prob >= 50.0

        # Confidence
        distance = abs(prob - 0.5)
        if distance >= 0.3:
            confidence = "High"
        elif distance >= 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        risk_factors = self._identify_risks(student_data, features.iloc[0])
        recommendations = self._generate_recommendations(
            student_data, features.iloc[0], risk_factors
        )

        return {
            "pass_probability": pass_prob,
            "will_pass": will_pass,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
        }

    # Batch prediction

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of students.

        Returns the original DataFrame with three new columns:
          pass_probability, will_pass, confidence
        """
        self._ensure_loaded()

        features = self.feature_engineer.transform(df)
        probs = self.model.predict_proba(features)[:, 1]

        result = df.copy()
        result["pass_probability"] = (probs * 100).round(2)
        result["will_pass"] = result["pass_probability"] >= 50.0

        def _conf(p):
            d = abs(p / 100 - 0.5)
            if d >= 0.3:
                return "High"
            elif d >= 0.15:
                return "Medium"
            return "Low"

        result["confidence"] = result["pass_probability"].apply(_conf)
        return result

    # Explanation

    def explain_prediction(self, student_data: dict) -> str:
        """
        Return a human-readable paragraph explaining the prediction.
        """
        pred = self.predict_single(student_data)
        df = pd.DataFrame([student_data])
        feats = self.feature_engineer.transform(df).iloc[0]

        skill = student_data.get("student_skill_level", "?")
        diff = student_data.get("difficulty_level", "?")
        days = student_data.get("days_since_last_study", "?")
        sessions = student_data.get("study_sessions_count", "?")
        prob = pred["pass_probability"]
        verdict = "PASS" if pred["will_pass"] else "FAIL"

        lines = [
            f"The student (skill level {skill}) is attempting a difficulty-{diff} assessment.",
            f"Based on {sessions} study sessions and {days} days since last study, "
            f"the model predicts a {prob:.1f}% probability of passing (verdict: {verdict}, "
            f"confidence: {pred['confidence']}).",
        ]

        if feats["skill_difficulty_gap"] >= 3:
            lines.append(
                "The student's skill level substantially exceeds the assessment "
                "difficulty, which is a strong positive indicator."
            )
        elif feats["skill_difficulty_gap"] <= -2:
            lines.append(
                "The assessment difficulty significantly exceeds the student's "
                "skill level, posing a major challenge."
            )

        if feats["recency_weighted_score"] < 30:
            lines.append(
                "The recency-weighted score is low, suggesting that the student "
                "has not studied recently enough to retain material effectively."
            )

        if feats["accuracy_ratio"] < 0.5:
            lines.append(
                "Past accuracy on attempted questions is below 50%, indicating "
                "potential gaps in foundational knowledge."
            )

        if pred["risk_factors"]:
            lines.append(
                "Key risk factors: " + "; ".join(pred["risk_factors"]) + "."
            )

        if pred["recommendations"]:
            lines.append(
                "Recommendations: " + "; ".join(pred["recommendations"]) + "."
            )

        return " ".join(lines)

    # Internal helpers

    def _ensure_loaded(self):
        if self.model is None:
            self.load_model()

    @staticmethod
    def _identify_risks(raw: dict, feats: pd.Series) -> list:
        """Identify human-readable risk factors from feature values."""
        risks = []
        if feats.get("skill_difficulty_gap", 0) < 0:
            risks.append("Assessment difficulty exceeds skill level")
        if feats.get("accuracy_ratio", 1) < 0.5:
            risks.append("Low historical accuracy (< 50%)")
        if feats.get("learning_consistency", 0) > 1.0:
            risks.append("Low study consistency (high variance in study hours)")
        if raw.get("days_since_last_study", 0) > 14:
            risks.append("Long gap since last study session (> 14 days)")
        if feats.get("preparation_intensity", 0) < 5:
            risks.append("Low preparation intensity")
        if feats.get("recency_weighted_score", 100) < 25:
            risks.append("Very low recency-weighted score")
        return risks

    @staticmethod
    def _generate_recommendations(raw: dict, feats: pd.Series,
                                   risks: list) -> list:
        """Generate actionable recommendations based on identified risks."""
        recs = []
        if "Low study consistency" in str(risks):
            recs.append("Maintain a regular study schedule with consistent "
                        "daily hours")
        if "Long gap since last study" in str(risks):
            recs.append("Review material within 3 days before the assessment")
        if "Low historical accuracy" in str(risks):
            recs.append("Focus on practice problems to improve question "
                        "accuracy above 60%")
        if "difficulty exceeds skill" in str(risks).lower():
            recs.append("Spend extra time on foundational topics before "
                        "attempting advanced material")
        if "Low preparation intensity" in str(risks):
            recs.append("Increase study sessions to at least 5 with "
                        "1.5+ hours each")
        if raw.get("study_sessions_count", 0) < 3:
            recs.append("Study at least 2 more sessions before the "
                        "assessment")
        if not recs:
            recs.append("Continue current study habits -- performance "
                        "indicators are positive")
        return recs

# MAIN -- demonstrate predictions

if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    print("=" * 60)
    print("  Pass Probability Predictor -- Demo")
    print("=" * 60)

    predictor = PassPredictor()
    predictor.load_model()

    # ── Single prediction
    sample_student = {
        "subject": 0,                   # encoded Mathematics
        "difficulty_level": 4,
        "student_skill_level": 6,
        "time_spent_minutes": 55.0,
        "study_sessions_count": 4,
        "study_hours_mean": 1.8,
        "study_hours_variance": 0.35,
        "days_since_last_study": 3.0,
        "previous_scores_avg": 62.5,
        "previous_scores_trend": 1.2,
        "questions_attempted": 30,
        "questions_correct": 18,
        "assessment_score": 58.0,
    }

    print("\n── Single Prediction ──")
    result = predictor.predict_single(sample_student)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    print("\n── Explanation ──")
    explanation = predictor.explain_prediction(sample_student)
    print(f"  {explanation}")

    # ── Batch prediction 
    print("\n── Batch Prediction (10 random students) ──")
    df = load_data()
    df = clean_data(df)
    sample = df.sample(10, random_state=99)

    # Drop target so we don't leak
    feature_cols = [c for c in sample.columns
                    if c not in ["passed", "student_id", "assessment_id"]]
    batch_result = predictor.predict_batch(sample[feature_cols])

    display_cols = ["difficulty_level", "student_skill_level",
                    "assessment_score", "pass_probability",
                    "will_pass", "confidence"]
    print(batch_result[display_cols].to_string(index=False))

    print("\n[OK] Prediction demo complete.")
