"""
scripts/demo_client.py
A Python-based demo client for the PlaceMux Phase 3 Task 2 API.
Works on Windows without needing curl.

Usage:
    python scripts/demo_client.py
"""
import json
import urllib.request
import urllib.error

API = "http://127.0.0.1:8899"

PREDICT_PAYLOAD = {
    "candidate_id": "C00135",
    "job_id": "J0769",
    "cand_experience_yrs": 7.29,
    "cand_expected_salary": 11.97,
    "cand_region": "East",
    "cand_activity_score": 0.35,
    "cand_skills": "react|sap|tally",
    "job_min_exp": 3.40,
    "job_salary_offered": 6.76,
    "job_region": "East",
    "job_popularity": 5.94,
    "job_req_skills": "accounting|java|seo",
}


def get(path):
    try:
        with urllib.request.urlopen(f"{API}{path}") as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  [ERROR] Could not reach {API}{path}: {e}")
        print("  Is 'python src/serving/app.py' running in another terminal?")
        return None


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    print("=" * 60)
    print("PlaceMux Phase 3 Task 2 — Live Demo Client")
    print("=" * 60)

    # Step 1: Health check
    print("\n[1] Health check  ->  GET /health")
    health = get("/health")
    if health is None:
        return
    print(json.dumps(health, indent=2))

    # Step 2: Prediction with explanation
    print("\n[2] Prediction with plain-English explanation  ->  POST /predict")
    print("    Candidate: skills=react|sap|tally, exp=7.3yrs, region=East")
    print("    Job:       skills=accounting|java|seo, min_exp=3.4yrs, region=East")
    prediction = post("/predict", PREDICT_PAYLOAD)
    if prediction:
        print(json.dumps(prediction, indent=2))

    # Step 3: Live metrics
    print("\n[3] Monitoring metrics  ->  GET /metrics")
    metrics = get("/metrics")
    if metrics:
        print(json.dumps(metrics, indent=2))

    print("\n" + "=" * 60)
    print("Demo complete. See docs/WORKED_EXAMPLE.md for the full walkthrough.")
    print("=" * 60)


if __name__ == "__main__":
    main()
