"""
scoring/schema.py — Step 2: define and validate the input contract.
Exactly what features, in what shape, this interface expects — a
consumer sending a malformed input gets a specific, actionable pydantic
error, not a cryptic sklearn stack trace three layers down.
"""
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, create_model, ConfigDict

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "model_artifact"


def _load_feature_names() -> list:
    config_path = ARTIFACT_DIR / "serving_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot build input contract: {config_path} not found.")
    config = json.loads(config_path.read_text())
    features = config.get("feature_names")
    if not features:
        raise ValueError(f"'feature_names' missing or empty in {config_path}")
    return features


FEATURE_NAMES = _load_feature_names()

# Build the record schema DYNAMICALLY from the packaged model's own
# feature_names — the input contract can never silently drift out of
# sync with what the model actually expects, because it's generated
# from the same artifact at import time, not hand-typed separately.
_field_definitions = {
    name.replace(" ", "_"): (float, Field(..., description=f"Value for '{name}'"))
    for name in FEATURE_NAMES
}

PatientRecord = create_model(
    "PatientRecord",
    __config__=ConfigDict(extra="forbid"),  # reject unexpected fields loudly
    **_field_definitions,
)


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[PatientRecord] = Field(..., min_length=1, description="One or more patient records to score")


def record_to_feature_dict(record) -> dict:
    """Undo the space->underscore mapping to get back to the exact
    column names the model's preprocessing pipeline was trained on."""
    raw = record.model_dump()
    return {name: raw[name.replace(" ", "_")] for name in FEATURE_NAMES}
