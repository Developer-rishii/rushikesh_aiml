import pandas as pd
import numpy as np
import os

def generate_datasets(students_count=1000, jobs_count=50, output_dir="data"):
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)  # For reproducibility

    # 1. Generate Students
    students_data = {
        "student_id": range(1, students_count + 1),
        # Assuming scores are out of 100
        "python": np.clip(np.random.normal(70, 15, students_count), 0, 100).astype(int),
        "machine_learning": np.clip(np.random.normal(60, 20, students_count), 0, 100).astype(int),
        "sql": np.clip(np.random.normal(75, 10, students_count), 0, 100).astype(int),
        "data_analysis": np.clip(np.random.normal(65, 18, students_count), 0, 100).astype(int),
        "communication": np.clip(np.random.normal(80, 12, students_count), 0, 100).astype(int),
    }
    students_df = pd.DataFrame(students_data)
    students_file = os.path.join(output_dir, "students.csv")
    students_df.to_csv(students_file, index=False)
    print(f"Generated {students_count} students: {students_file}")

    # 2. Generate Jobs
    jobs_data = {
        "job_id": range(1, jobs_count + 1),
        # Job thresholds tend to be varied. Some require high skills, some low.
        "python_threshold": np.clip(np.random.normal(65, 15, jobs_count), 0, 100).astype(int),
        "machine_learning_threshold": np.clip(np.random.normal(55, 20, jobs_count), 0, 100).astype(int),
        "sql_threshold": np.clip(np.random.normal(60, 15, jobs_count), 0, 100).astype(int),
        "data_analysis_threshold": np.clip(np.random.normal(60, 15, jobs_count), 0, 100).astype(int),
        "communication_threshold": np.clip(np.random.normal(70, 15, jobs_count), 0, 100).astype(int),
    }
    jobs_df = pd.DataFrame(jobs_data)
    jobs_file = os.path.join(output_dir, "jobs.csv")
    jobs_df.to_csv(jobs_file, index=False)
    print(f"Generated {jobs_count} jobs: {jobs_file}")

if __name__ == "__main__":
    # If run as a script, generate data in the parent's data/ folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    generate_datasets(output_dir=data_dir)
