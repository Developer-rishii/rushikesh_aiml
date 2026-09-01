"""
schemas.py
----------
Defines the strict input/output contract for the prediction endpoint using
pydantic. Any request that does not match this shape is rejected with a
422 before it ever reaches the model (garbage-input handling).
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

# Exact feature order the model was trained on (from sklearn's
# load_breast_cancer). The API validates against this so a caller cannot
# silently send features in the wrong order.
FEATURE_ORDER = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description=f"Exactly {len(FEATURE_ORDER)} numeric features, in "
        "the order defined by FEATURE_ORDER.",
    )

    @field_validator("features")
    @classmethod
    def validate_length_and_values(cls, v: List[float]) -> List[float]:
        if len(v) != len(FEATURE_ORDER):
            raise ValueError(
                f"Expected {len(FEATURE_ORDER)} features, got {len(v)}."
            )
        for i, val in enumerate(v):
            if val is None:
                raise ValueError(f"Feature at index {i} is null.")
            if isinstance(val, bool):
                raise ValueError(f"Feature at index {i} must be numeric, not bool.")
            if not isinstance(val, (int, float)):
                raise ValueError(f"Feature at index {i} must be numeric.")
            if val != val:  # NaN check
                raise ValueError(f"Feature at index {i} is NaN.")
            if abs(val) > 1e6:
                raise ValueError(
                    f"Feature at index {i} value {val} is out of a sane range."
                )
        return v


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
