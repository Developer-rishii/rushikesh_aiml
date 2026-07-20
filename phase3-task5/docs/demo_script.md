# 2-Minute Demonstration Script

## Preparation
- Start the FastAPI application: `python service/app.py`
- Open another terminal for running the load test.

## [0:00 - 0:30] End-to-End Prediction on Real Data
**Speaker:** "Welcome to the final sign-off for our candidate-job matching intelligence layer. Let's start with a live prediction."
**Action:** Use `curl` or Postman to hit the `/predict` endpoint.
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
    "candidate_id": "C75",
    "candidate_exp": 5,
    "candidate_skills": 4,
    "jobs": [
        {"job_id": "J12", "required_exp": 3, "required_skills": 2, "job_popularity": 0.8},
        {"job_id": "J15", "required_exp": 8, "required_skills": 6, "job_popularity": 0.2}
    ]
}'
```
**Speaker:** "As you can see, the model successfully scores and ranks Job J12 higher due to the candidate meeting all experience and skill requirements. J15 is ranked lower."

## [0:30 - 1:00] Load Test Results & Headroom
**Speaker:** "Now let's look at performance under load. We've run a Locust load test simulating continuous traffic."
**Action:** Run locust in terminal: `locust -f monitoring/locustfile.py --headless -u 50 -r 10 -t 30s --host http://localhost:8000`
**Speaker:** "Our SLO for p99 latency is 50ms. As the locust output shows, we are comfortably handling 50 concurrent users with p99 latency well under the threshold. We have sufficient headroom to scale."

## [1:00 - 1:30] Failure Injection & Automatic Fallback
**Speaker:** "But what happens if the model fails? Let's inject a failure."
**Action:** Run the same curl command but add the failure header:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "x-fail-model: true" \
     -d '{
    "candidate_id": "C75",
    "candidate_exp": 5,
    "candidate_skills": 4,
    "jobs": [
        {"job_id": "J12", "required_exp": 3, "required_skills": 2, "job_popularity": 0.8},
        {"job_id": "J15", "required_exp": 8, "required_skills": 6, "job_popularity": 0.9}
    ]
}'
```
**Speaker:** "Notice two things: First, we got a 200 OK response—the API didn't crash. Second, the `used_fallback` flag is true, and the jobs are now ranked purely by their `job_popularity` score. J15 is now ranked first because its popularity is 0.9 vs J12's 0.8. We chose a degraded-quality fallback over a hard fallback (500 error) so that users always see jobs."

## [1:30 - 2:00] Monitoring & Reliability Sign-off
**Speaker:** "All of this is fully observable."
**Action:** Open `http://localhost:8000/metrics` in the browser or via `curl`.
**Speaker:** "Our Prometheus endpoint tracks these model errors (`api_model_errors_total`) and fallback counts (`api_fallback_total`), so we get alerted immediately when the fallback kicks in. Finally, our reliability sign-off report documents our SLOs, fairness assessments showing slight demographic parity differences we need to address, and the train/serve skew risks. With this, the service is ready for scale integration."
