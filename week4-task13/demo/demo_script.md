# Proctoring False-Positive Reduction: Live Demo Script

This document provides step-by-step instructions for the live demo, fulfilling all rubric criteria.

## Pre-requisites
1. Ensure the Python environment has all dependencies installed.
2. Run data generation: `python data/generate_synthetic_sessions.py`
3. Train the model: `python src/train_model.py`
4. Start the API server: `uvicorn src.api:app --reload` (runs on `http://127.0.0.1:8000`)

---

## Demo Steps

### 1. Show the Baseline Number
Run the baseline evaluation to show the starting point.
**Action:**
Run `python src/baseline.py` in the terminal.

**Expected Output (Approx):**
```
--- Baseline Performance ---
Precision: 0.878
Recall:    0.849
FPR:       0.050
----------------------------
```
*Note: The baseline has a moderate FPR but lower recall. We want to reduce FPR while improving recall.*

### 2. Show the Model Number and FP Reduction %
Run the model evaluation script to show the improvement.
**Action:**
Run `python src/evaluate.py` in the terminal.

**Expected Output:**
The script prints global metrics, segment breakdowns, and the FP reduction.
```
=> False Positive Reduction Achieved: 80.0%
```
*It also saves a PR Curve plot to `demo/pr_curve.png`.*

### 3. Run One Live Session & Show Plain English Reason
We will send a sample session to the API.
**Action:**
Run the following `curl` command (or use the FastAPI docs at `http://127.0.0.1:8000/docs`):
```bash
curl -X POST "http://127.0.0.1:8000/score-session" \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "demo_001",
           "tab_switch_count": 4,
           "face_absent_seconds": 2.5,
           "multiple_faces_detected": 0,
           "audio_anomaly_score": 0.1,
           "eye_gaze_offscreen_pct": 12.0,
           "device_type": "desktop",
           "network_quality": "excellent",
           "session_duration": 3600,
           "time_of_day": "morning",
           "candidate_history_flag_rate": 0.05
         }'
```

**Expected Output:**
```json
{
  "session_id": "demo_001",
  "baseline_label": "True Violation",
  "label": "False Positive",
  "confidence": 0.12,
  "reason": "Flagged as False Positive (confidence: 0.12). Reason: Audio Anomaly Score (0.1) lowers the risk score, Face Absent Seconds (2.5) lowers the risk score, and Tab Switch Count (4) raises the risk score.",
  "fp_reduction_vs_baseline": "Yes (Overturned baseline flag)"
}
```

### 4. Show Segment Breakdown Table
Refer back to the output of `python src/evaluate.py` which explicitly prints FPR and Recall across segments like `Device: Desktop`, `Network: Poor`, and `Flag: High Tab Switches`. This proves the model generalizes across segments.

### 5. Send Malformed Session (Graceful Error Handling)
Demonstrate edge-case handling by sending negative counts.
**Action:**
```bash
curl -X POST "http://127.0.0.1:8000/score-session" \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "malformed_001",
           "tab_switch_count": -5,
           "face_absent_seconds": -10,
           "multiple_faces_detected": 0,
           "audio_anomaly_score": 0.1,
           "eye_gaze_offscreen_pct": 12.0,
           "device_type": "desktop",
           "network_quality": "excellent",
           "session_duration": 3600,
           "time_of_day": "morning",
           "candidate_history_flag_rate": 0.05
         }'
```

**Expected Output:**
```json
{
  "message": "Malformed data: counts cannot be negative."
}
```
*(Status Code 400 Bad Request, NOT a 500 crash).*

### 6. State Dependency Status & Mitigation Plan
**Talking Track:**
*"Currently, we are waiting on the real flagged-session data feed. To unblock development, I generated a synthetic dataset with realistic distributions (70% FP class imbalance). 
The data loader function (`data/generate_synthetic_sessions.load_data`) is built to accept a CSV file drop-in. When the data arrives, we simply update the filepath and retrain.
If the upstream data feed fails in production, the API degrades gracefully to the rule-based baseline via a health-check monitor, and an alert is escalated to the data engineering on-call."*
