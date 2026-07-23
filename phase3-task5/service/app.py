"""
FastAPI Candidate-Job Matching Service.

Design decisions:
- Startup event: load_model() is called at startup and raises if it fails,
  so a broken deployment never silently serves as "healthy."
- /health: returns model_loaded: true/false and non-200 if model isn't loaded.
- Fallback reasons: Prometheus counter label distinguishes "forced" (injection),
  "model_error" (runtime exception), and "startup_not_loaded" (model never loaded),
  so an on-call engineer can tell the difference in dashboards.
- Exception handling: the broad except is kept as a last-resort safety net,
  but now logs the fallback_reason distinctly rather than swallowing silently.
"""
import time
import sys
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Header
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from model_loader import load_model, is_model_loaded, get_model_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("service")

# --- Lifespan (startup/shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup. Fail loud if model can't load."""
    logger.info("Starting up: loading model...")
    try:
        load_model()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.critical(f"FATAL: Model failed to load at startup: {e}")
        sys.exit(1)
    yield
    logger.info("Shutting down.")

app = FastAPI(title="Candidate Job Matching API", lifespan=lifespan)

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    'api_requests_total', 'Total number of API requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds', 'Request latency in seconds',
    ['endpoint']
)
FALLBACK_COUNT = Counter(
    'api_fallback_total', 'Total number of predictions using fallback',
    ['fallback_reason']
)
MODEL_ERROR_COUNT = Counter(
    'api_model_errors_total', 'Total number of model execution errors',
    ['error_type']
)

# --- Middleware for metrics ---
@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()
    if request.url.path == "/predict":
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

    return response

# --- Request/Response Models ---
class JobFeatures(BaseModel):
    job_id: str
    required_exp: int
    required_skills: int
    job_popularity: float

class PredictRequest(BaseModel):
    candidate_id: str
    candidate_exp: int
    candidate_skills: int
    jobs: List[JobFeatures]

class RankedJob(BaseModel):
    job_id: str
    score: float

class PredictResponse(BaseModel):
    candidate_id: str
    ranked_jobs: List[RankedJob]
    used_fallback: bool
    fallback_reason: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None

# --- Endpoints ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    loaded = is_model_loaded()
    status_code = 200 if loaded else 503
    return Response(
        content='{"status": "ok", "model_loaded": true}' if loaded
        else '{"status": "degraded", "model_loaded": false}',
        media_type="application/json",
        status_code=status_code
    )

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, x_fail_model: Optional[str] = Header(None)):
    used_fallback = False
    fallback_reason = None
    ranked = []
    meta = get_model_metadata()
    model_version = meta.get("model_version", "1.0.0")
    run_id = meta.get("run_id", "unknown")

    # Check for forced failure (failure injection)
    if x_fail_model == "true":
        used_fallback = True
        fallback_reason = "forced"
        MODEL_ERROR_COUNT.labels(error_type="forced_injection").inc()
        logger.info(f"Failure injection triggered for candidate {request.candidate_id}")
    elif not is_model_loaded():
        used_fallback = True
        fallback_reason = "startup_not_loaded"
        MODEL_ERROR_COUNT.labels(error_type="startup_not_loaded").inc()
        logger.warning("Model not loaded, using fallback")
    else:
        try:
            model = load_model()

            # Prepare features
            features = []
            job_ids = []
            for job in request.jobs:
                features.append([
                    request.candidate_exp,
                    request.candidate_skills,
                    job.required_exp,
                    job.required_skills,
                    job.job_popularity
                ])
                job_ids.append(job.job_id)

            # Predict
            df_features = pd.DataFrame(
                features,
                columns=['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
            )
            scores = model.predict(df_features)

            # Rank
            scored_jobs = list(zip(job_ids, scores))
            scored_jobs.sort(key=lambda x: x[1], reverse=True)

            ranked = [RankedJob(job_id=jid, score=float(score)) for jid, score in scored_jobs]

        except Exception as e:
            logger.error(f"Model prediction failed (model_error): {e}", exc_info=True)
            MODEL_ERROR_COUNT.labels(error_type="model_error").inc()
            used_fallback = True
            fallback_reason = "model_error"

    if used_fallback:
        FALLBACK_COUNT.labels(fallback_reason=fallback_reason).inc()
        # Fallback Strategy: Rank by job popularity (highest first)
        sorted_jobs = sorted(request.jobs, key=lambda x: x.job_popularity, reverse=True)
        ranked = [RankedJob(job_id=job.job_id, score=job.job_popularity) for job in sorted_jobs]

    return PredictResponse(
        candidate_id=request.candidate_id,
        ranked_jobs=ranked,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        model_version=model_version,
        run_id=run_id
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
