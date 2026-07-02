"""
FastAPI application for the Skills Ontology Mapper.

Endpoints:
  POST /map-skills          — Map a list of raw parsed skill strings to canonical nodes
  GET  /match-preview/{id}  — Show before/after of raw skills vs ontology-tagged skills
  GET  /health              — Health check

Edge-case handling:
  - Empty input list → 400
  - Missing 'raw_terms' field → 400
  - Extremely long strings → truncated to 500 chars
  - Duplicate terms → deduplicated in results
  - All unmappable → valid response with all items in unmapped list
"""

import os
import sys
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(__file__))
from mapper import SkillMapper
from explain import generate_reason

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skills_ontology_api")

# ---------------------------------------------------------------------------
# App & Mapper
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PlaceMux Skills Ontology Mapper",
    description="Maps raw parsed skill strings to canonical ontology nodes. "
                "Part of the Parsing -> Skills Ontology pipeline for PlaceMux.",
    version="1.0.0",
)

# Load mapper once at startup
mapper = SkillMapper()

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

MAX_RAW_TERM_LENGTH = 500
MAX_BATCH_SIZE = 1000


class MapSkillsRequest(BaseModel):
    raw_terms: list[str] = Field(
        ...,
        description="List of raw parsed skill strings from Parsing v0",
        min_length=1,
    )

    @field_validator("raw_terms", mode="before")
    @classmethod
    def validate_raw_terms(cls, v):
        if not isinstance(v, list):
            raise ValueError("raw_terms must be a list of strings")
        if len(v) == 0:
            raise ValueError("raw_terms must not be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
        # Truncate extremely long strings
        return [
            str(t)[:MAX_RAW_TERM_LENGTH] if isinstance(t, str) and len(t) > MAX_RAW_TERM_LENGTH
            else str(t) if t is not None else ""
            for t in v
        ]


class MappedSkill(BaseModel):
    raw: str
    canonical_id: str
    display_name: Optional[str]
    confidence: float
    reason: str
    method: str


class MapSkillsResponse(BaseModel):
    mapped: list[MappedSkill]
    unmapped: list[MappedSkill]
    total_input: int
    total_mapped: int
    total_unmapped: int


class MatchPreviewResponse(BaseModel):
    student_id: str
    raw_skills: list[str]
    ontology_tagged: list[dict]
    unmapped_terms: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Simulated student data (for match-preview demo)
# ---------------------------------------------------------------------------

SIMULATED_STUDENTS = {
    "stu_001": {
        "name": "Aarav Sharma",
        "raw_parsed_skills": [
            "Python (3 yrs)", "ReactJS", "Sr. Java Developer",
            "machine learning", "K8s", "postgress", "agile",
            "Docker Compose", "TensorFlow", "skills:", "flutter",
        ],
    },
    "stu_002": {
        "name": "Priya Patel",
        "raw_parsed_skills": [
            "javascript", "vue.js", "NodeJS", "MongoDB",
            "HTML5", "CSS3", "Git", "REST API",
            "pytohn", "xkq7z", "team player",
        ],
    },
    "stu_003": {
        "name": "Raj Kapoor",
        "raw_parsed_skills": [
            "Sr. Python Dev", "ML", "deep learning", "pytorrch",
            "scikit learn", "pandas", "numpy", "sql",
            "data science", "problem solving", "•", "N/A",
        ],
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to the interactive API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ontology_size": len(mapper.ontology),
        "service": "skills-ontology-mapper",
        "version": "1.0.0",
    }


@app.post("/map-skills", response_model=MapSkillsResponse)
def map_skills(request: MapSkillsRequest):
    """
    Map a list of raw parsed skill strings to canonical ontology nodes.

    Each term is mapped through three layers:
    1. Exact/synonym lookup (confidence=1.0)
    2. Fuzzy string match (confidence 0.5-0.95)
    3. TF-IDF cosine similarity (confidence 0.4-0.9)

    Terms below all thresholds are returned as 'unmapped'.
    """
    logger.info(f"Received {len(request.raw_terms)} terms for mapping")

    results = mapper.map_batch(request.raw_terms)

    mapped = []
    unmapped = []

    for result in results:
        reason = generate_reason(result)
        skill = MappedSkill(
            raw=result["raw"],
            canonical_id=result["canonical_id"],
            display_name=result.get("display_name"),
            confidence=result["confidence"],
            reason=reason,
            method=result.get("method", "unknown"),
        )

        if result["canonical_id"] == "unmapped":
            unmapped.append(skill)
        else:
            mapped.append(skill)

    response = MapSkillsResponse(
        mapped=mapped,
        unmapped=unmapped,
        total_input=len(request.raw_terms),
        total_mapped=len(mapped),
        total_unmapped=len(unmapped),
    )

    logger.info(f"Mapped {len(mapped)}/{len(request.raw_terms)} terms successfully")
    return response


@app.get("/match-preview/{student_id}", response_model=MatchPreviewResponse)
def match_preview(student_id: str):
    """
    Show the before/after of a student's raw skills vs ontology-tagged skills.

    This endpoint demonstrates the 'richer matching inputs' hand-off:
    raw noisy skill list -> clean ontology-tagged list used for matching.

    Uses simulated student data (would connect to real Parsing v0 in production).
    """
    if student_id not in SIMULATED_STUDENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Student '{student_id}' not found. "
                   f"Available IDs: {list(SIMULATED_STUDENTS.keys())}",
        )

    student = SIMULATED_STUDENTS[student_id]
    raw_skills = student["raw_parsed_skills"]

    # Map all skills through the ontology mapper
    results = mapper.map_batch(raw_skills)

    ontology_tagged = []
    unmapped_terms = []

    for result in results:
        reason = generate_reason(result)
        if result["canonical_id"] != "unmapped":
            ontology_tagged.append({
                "canonical_id": result["canonical_id"],
                "display_name": result["display_name"],
                "confidence": result["confidence"],
                "source_raw": result["raw"],
                "reason": reason,
            })
        else:
            unmapped_terms.append(result["raw"])

    # Deduplicate tagged skills by canonical_id (keep highest confidence)
    seen = {}
    for tag in ontology_tagged:
        cid = tag["canonical_id"]
        if cid not in seen or tag["confidence"] > seen[cid]["confidence"]:
            seen[cid] = tag
    ontology_tagged_dedup = list(seen.values())

    summary = (
        f"Student '{student['name']}' ({student_id}): "
        f"{len(raw_skills)} raw terms -> "
        f"{len(ontology_tagged_dedup)} unique canonical skills, "
        f"{len(unmapped_terms)} unmapped. "
        f"These {len(ontology_tagged_dedup)} clean skills feed the matching engine."
    )

    return MatchPreviewResponse(
        student_id=student_id,
        raw_skills=raw_skills,
        ontology_tagged=ontology_tagged_dedup,
        unmapped_terms=unmapped_terms,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Custom error handlers
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Return 400 instead of 422 for validation errors (cleaner API)."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "hint": "Ensure 'raw_terms' is a non-empty list of strings.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all to prevent 500s from leaking."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Starting PlaceMux Skills Ontology Mapper API")
    print(f"Ontology: {len(mapper.ontology)} canonical skills loaded")
    print("Endpoints:")
    print("  POST /map-skills          - Map raw skill strings")
    print("  GET  /match-preview/{id}  - Before/after match preview")
    print("  GET  /health              - Health check")
    print("  GET  /docs                - Interactive API docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
