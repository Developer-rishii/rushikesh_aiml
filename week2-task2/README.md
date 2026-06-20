# Job Matching System

Phase 2 implementation for the Job Matching System. This system uses baseline techniques to match candidates with jobs by generating match vectors, validating skill thresholds, and computing match scores.

## Project Structure

```
week2-task2/
├── data/                    # Generated sample data
│   ├── students.csv
│   └── jobs.csv
├── src/                     # Core matching logic
│   ├── load_data.py         # Data generation
│   ├── match_vector.py      # Match vector generator
│   ├── threshold_validator.py# Validation logic
│   ├── scoring.py           # Match score calculator
│   ├── explainability.py    # Human-readable explanations
│   └── evaluation.py        # Metrics calculations
├── notebooks/               # Jupyter notebooks
│   └── experiments.ipynb
├── api/                     # FastAPI application
│   └── app.py
├── logs/                    # Experiment logs
│   └── experiments.csv
├── tests/                   # Pytest suite
│   └── test_matching.py
├── demo.py                  # CLI demo script
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate data:
   ```bash
   python src/load_data.py
   ```

## Matching Logic

The system follows a straightforward, rules-based logic:
1. **Match Vector**: Creates a binary vector where `1` indicates the student's skill meets or exceeds the job's required threshold, and `0` indicates it does not.
2. **Threshold Validation**: Checks if all mandatory job skill thresholds are met by the student. Returns boolean eligibility and missing skills.
3. **Match Score**: Calculates the percentage of required skills met by the student.
4. **Explainability**: Generates a detailed text explanation outlining exactly which skills were met and which fell short.

## Running the Demo

To run an end-to-end matching demo on random examples from the dataset:

```bash
python demo.py
```

## API Usage

1. Start the FastAPI server:
   ```bash
   uvicorn api.app:app --reload
   ```

2. Send a POST request to the `/match` endpoint:
   ```json
   POST http://localhost:8000/match
   {
       "student_id": 1,
       "job_id": 3
   }
   ```

## Evaluation

Run the evaluation script to calculate Precision, Recall, and False Positive Rate on sample data (saved to `logs/experiments.csv`):

```bash
python src/evaluation.py
```
Or explore the metrics interactively via the Jupyter notebook in `notebooks/experiments.ipynb`.
