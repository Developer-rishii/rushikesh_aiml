#!/bin/bash
# Demo script for Rec v1
# Run the FastAPI server in another terminal first:
# uvicorn api.main:app --host 0.0.0.0 --port 8000

echo "=== 1. Live Recommendation (One Example Walkthrough) ==="
echo "Fetching top-5 ranked jobs for student_A_0 in college_A..."
curl -s http://localhost:8000/recommend/college_A/student_A_0 | jq .
echo -e "\n"

echo "=== 2. Explainability (Baseline vs Model) ==="
echo "Comparing model vs baseline for student_A_0..."
curl -s http://localhost:8000/recommend/college_A/student_A_0/explain | jq .
echo -e "\n"

echo "=== 3. Data Isolation Proof (Cross-College Attempt) ==="
echo "Attempting to fetch college_B's student (student_B_0) using college_A's endpoint..."
curl -s -w "\nHTTP Status: %{http_code}\n" http://localhost:8000/recommend/college_A/student_B_0
echo -e "\n"

echo "=== 4. College Portal Dashboard ==="
echo "Fetching dashboard for college_B (Placement Officer view)..."
curl -s http://localhost:8000/portal/college_B/dashboard | jq .
echo -e "\n"

echo "=== 5. Edge Cases API Check ==="
echo "Fetching edge cases summary..."
curl -s http://localhost:8000/rec/edge-cases | jq .
echo -e "\n"

echo "=== 6. Full Metrics Report Check ==="
echo "Fetching first few lines of the full metrics report..."
curl -s http://localhost:8000/rec/report | jq . | head -n 25
echo -e "\n"
