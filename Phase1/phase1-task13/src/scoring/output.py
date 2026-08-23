"""
scoring/output.py — Step 3: standardise the output (score + meaning +
model version). Every single score this interface ever returns — batch
or single-record — goes through this one Score object, so the output
shape can never drift between call sites.
"""
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class Score(BaseModel):
    """One fully self-explanatory prediction. A non-ML engineer reading
    this JSON should not need to ask 'what does 0.82 mean' — it's answered
    inline in every field, per the brainstorming question "Is the score
    self-explanatory to a non-ML engineer?"."""
    model_config = ConfigDict(protected_namespaces=())
    probability: float = Field(..., ge=0.0, le=1.0,
                                description="Calibrated probability of the positive class (benign)")
    score_meaning: str = Field(..., description="What the probability means and how it was produced")
    decision: int = Field(..., description="0 or 1: final decision after applying the operating threshold")
    decision_label: str = Field(..., description="Human-readable label for the decision")
    threshold_used: float = Field(..., description="Operating threshold applied to reach the decision")
    model_version: str = Field(..., description="Identifier for exactly which trained model produced this score")
    model_name: str = Field(..., description="Model family/algorithm name")
    calibration_method: str = Field(..., description="Probability calibration method used (Step 2, Task 12)")


class BatchScoreResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    scores: List[Score]
    n_records: int
    model_version: str
