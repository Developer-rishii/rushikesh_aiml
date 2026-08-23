"""
scoring/interface.py — Steps 1, 3, 4, 5: the clean predict interface.
This is the single entrypoint any downstream system calls — it wraps
model loading, input validation, scoring, output standardisation, and
graceful error handling all in one place, so a consumer never touches
the raw sklearn pipeline directly.
"""
import logging
import pandas as pd
from pydantic import ValidationError

from src.scoring.schema import PatientRecord, BatchScoreRequest, record_to_feature_dict, FEATURE_NAMES
from src.scoring.output import Score, BatchScoreResult
from src.scoring.registry import compute_model_version, load_model_metadata, MODEL_PATH

log = logging.getLogger("src.scoring.interface")


class ScoringError(Exception):
    """Raised for any input or scoring failure — a single, well-defined
    exception type a consumer can catch, instead of leaking sklearn/
    pydantic internals up the call stack."""
    def __init__(self, message: str, detail: dict = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class Scorer:
    """Step 1: the clean predict interface. Loads the model ONCE at
    construction; every .score_one()/.score_batch() call reuses it."""

    def __init__(self):
        import joblib
        if not MODEL_PATH.exists():
            raise ScoringError(f"Model artifact not found at {MODEL_PATH}. "
                                f"Hand-off from Task 12's serving package was not completed.")
        self.model = joblib.load(MODEL_PATH)
        self.metadata = load_model_metadata()
        self.threshold = self.metadata["threshold"]
        self.calibration_method = self.metadata.get("calibration_method", "unknown")
        self.model_name = self.metadata.get("model_name", "unknown")
        self.model_version = compute_model_version()
        log.info("Scorer initialised: model_version=%s, threshold=%s, calibration=%s",
                  self.model_version, self.threshold, self.calibration_method)

    def _score_meaning(self) -> str:
        return (
            f"Calibrated ({self.calibration_method}) probability that the tumor is BENIGN "
            f"(positive class = 1). A value near 1.0 means high confidence of benign; near 0.0 "
            f"means high confidence of malignant. The 'decision' field applies the operating "
            f"threshold ({self.threshold}) chosen in Task 12 to minimize the real clinical cost "
            f"of errors — it is NOT a plain 0.5 cutoff."
        )

    def _build_score(self, probability: float) -> Score:
        decision = int(probability >= self.threshold)
        return Score(
            probability=round(float(probability), 6),
            score_meaning=self._score_meaning(),
            decision=decision,
            decision_label="benign" if decision == 1 else "malignant",
            threshold_used=self.threshold,
            model_version=self.model_version,
            model_name=self.model_name,
            calibration_method=self.calibration_method,
        )

    # ---- Step 4: single-record scoring ----
    def score_one(self, record: dict) -> Score:
        """Step 2/5: validate a single raw dict against the input
        contract; raise a well-defined ScoringError on malformed input
        instead of letting a KeyError/sklearn error propagate raw."""
        try:
            validated = PatientRecord(**record)
        except ValidationError as e:
            raise ScoringError("Input validation failed for single record.",
                                detail={"pydantic_errors": e.errors()}) from e

        feature_dict = record_to_feature_dict(validated)
        X = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
        try:
            proba = self.model.predict_proba(X)[0, 1]
        except Exception as e:
            raise ScoringError(f"Model inference failed: {e}") from e
        return self._build_score(proba)

    # ---- Step 4: batch scoring ----
    def score_batch(self, records: list) -> BatchScoreResult:
        try:
            validated_request = BatchScoreRequest(records=records)
        except ValidationError as e:
            raise ScoringError("Input validation failed for batch request.",
                                detail={"pydantic_errors": e.errors()}) from e

        feature_dicts = [record_to_feature_dict(r) for r in validated_request.records]
        X = pd.DataFrame(feature_dicts, columns=FEATURE_NAMES)
        try:
            probas = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            raise ScoringError(f"Batch model inference failed: {e}") from e

        scores = [self._build_score(p) for p in probas]
        return BatchScoreResult(scores=scores, n_records=len(scores), model_version=self.model_version)
