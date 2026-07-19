import requests
import json

API_URL = "http://127.0.0.1:8000"

def check_health():
    print(f"--- Checking {API_URL}/health ---")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict():
    print(f"--- Testing {API_URL}/predict ---")
    payload = {
        "skill_overlap_score": 0.85,
        "years_experience": 5.0,
        "experience_gap": 0.0,
        "resume_parse_confidence": 0.95,
        "interview_eval_score": 0.88,
        "communication_score": 0.92,
        "role_historical_hire_rate": 0.15
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    try:
        check_health()
        test_predict()
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Is the API running at {API_URL}?")
        print("Run 'uvicorn src.api.main:app --port 8000' in another terminal first.")
