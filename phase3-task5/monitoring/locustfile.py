from locust import HttpUser, task, between
import random
import uuid

class CandidateMatcherUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(4)
    def normal_prediction(self):
        # Prepare a random request
        candidate_id = f"C{random.randint(1, 1000)}"
        payload = {
            "candidate_id": candidate_id,
            "candidate_exp": random.randint(1, 15),
            "candidate_skills": random.randint(1, 10),
            "jobs": [
                {
                    "job_id": f"J{random.randint(1, 500)}",
                    "required_exp": random.randint(1, 10),
                    "required_skills": random.randint(1, 10),
                    "job_popularity": random.uniform(0.1, 1.0)
                } for _ in range(5) # Ask to rank 5 jobs
            ]
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("used_fallback"):
                    response.success() # Fallback is still a success for SLOs
                else:
                    response.success()
            else:
                response.failure(f"Got {response.status_code}")
                
    @task(1)
    def failure_injected_prediction(self):
        # 20% of traffic will have the failure header injected
        candidate_id = f"C{random.randint(1, 1000)}"
        payload = {
            "candidate_id": candidate_id,
            "candidate_exp": random.randint(1, 15),
            "candidate_skills": random.randint(1, 10),
            "jobs": [
                {
                    "job_id": f"J{random.randint(1, 500)}",
                    "required_exp": random.randint(1, 10),
                    "required_skills": random.randint(1, 10),
                    "job_popularity": random.uniform(0.1, 1.0)
                } for _ in range(5)
            ]
        }
        
        headers = {"x-fail-model": "true"}
        
        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("used_fallback"):
                    response.success() # We expect fallback
                else:
                    response.failure("Expected fallback, but model succeeded")
            else:
                response.failure(f"Got {response.status_code}")
