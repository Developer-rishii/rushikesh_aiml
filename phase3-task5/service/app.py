import time
import pandas as pd
from fastapi import FastAPI, Request, Response, Header
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from model_loader import load_model

app = FastAPI(title="Candidate Job Matching API")

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter('api_requests_total', 'Total number of API requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
FALLBACK_COUNT = Counter('api_fallback_total', 'Total number of predictions using fallback')
MODEL_ERROR_COUNT = Counter('api_model_errors_total', 'Total number of model execution errors')

# --- Middleware for metrics ---
@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    if request.url.path == "/predict":
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
        
    return response

# --- Models ---
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

# --- Endpoints ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, x_fail_model: Optional[str] = Header(None)):
    used_fallback = False
    ranked = []
    
    # Check for forced failure (failure injection)
    if x_fail_model == "true":
        used_fallback = True
        MODEL_ERROR_COUNT.inc()
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
            df_features = pd.DataFrame(features, columns=['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity'])
            scores = model.predict(df_features)
            
            # Rank
            scored_jobs = list(zip(job_ids, scores))
            scored_jobs.sort(key=lambda x: x[1], reverse=True)
            
            ranked = [RankedJob(job_id=jid, score=float(score)) for jid, score in scored_jobs]
            
        except Exception as e:
            print(f"Model prediction failed: {e}")
            MODEL_ERROR_COUNT.inc()
            used_fallback = True
            
    if used_fallback:
        FALLBACK_COUNT.inc()
        # Fallback Strategy: Rank by job popularity (highest first)
        sorted_jobs = sorted(request.jobs, key=lambda x: x.job_popularity, reverse=True)
        ranked = [RankedJob(job_id=job.job_id, score=job.job_popularity) for job in sorted_jobs]
        
    return PredictResponse(
        candidate_id=request.candidate_id,
        ranked_jobs=ranked,
        used_fallback=used_fallback
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
