"""
Stage E: "Show recommendations for a real user with the 'why', plus the offline
metrics." This produces the exact artifact a 2-minute live demo would present.
"""
import os, sys, json, glob
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recommender import recommend_for_candidate, recommend_candidates_for_job

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "model_registry")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "experiments", "experiment_log.json")


def main():
    with open(LOG_PATH) as f:
        entry = json.load(f)[-1]
    model = joblib.load(entry["model_path"])
    candidates = pd.read_csv(os.path.join(DATA_DIR, "candidates.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))

    demo_candidate = candidates.iloc[3]["candidate_id"]
    demo_job = jobs.iloc[7]["job_id"]

    candidate_to_jobs = recommend_for_candidate(model, demo_candidate, candidates, jobs, k=5)
    job_to_candidates = recommend_candidates_for_job(model, demo_job, candidates, jobs, k=5)

    eval_file = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                               "offline_eval_*.json")))[-1]
    latency_file = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                                  "serving_latency_*.json")))[-1]
    with open(eval_file) as f:
        offline_eval = json.load(f)
    with open(latency_file) as f:
        latency = json.load(f)

    demo_output = {
        "model_version": entry["version"],
        "candidate_to_jobs_example": {
            "candidate_id": demo_candidate,
            "top_5_recommendations": candidate_to_jobs,
        },
        "job_to_candidates_example": {
            "job_id": demo_job,
            "top_5_candidates": job_to_candidates,
        },
        "offline_eval_vs_baseline": offline_eval,
        "serving_latency_slo": latency,
    }
    out_path = os.path.join(os.path.dirname(__file__), "demo_output.json")
    with open(out_path, "w") as f:
        json.dump(demo_output, f, indent=2)
    print(json.dumps(demo_output, indent=2))


if __name__ == "__main__":
    main()
