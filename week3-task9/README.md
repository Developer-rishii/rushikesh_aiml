# PlaceMux: Conversion-Quality Check

This repository contains the "Conversion-Quality Check" deliverable for the PlaceMux matching system. PlaceMux operates a paywall where students pay to unlock job matches. This check exists to ensure that the payment system (and failures around it) do not silently skew match relevance.

## The Conversion-Quality Check Metric
The core metric is the **segmented precision and recall** of the matching system, divided by payment status (`paid`, `failed`, `pending`, `refunded`). 

**Why it matters:** If an unpaid student (whose payment failed or is pending) with identical skills to a paid student receives significantly worse matches (lower precision/recall), the paywall has skewed the relevance of the matching pool. 

**The Threshold:** The check computes the baseline precision of a simple skill-overlap heuristic. It then compares each segment's precision against this baseline. If any segment drops below the baseline precision by more than a configurable threshold (e.g., 5 percentage points), a **relevance regression** is flagged. This run history is persisted so regressions can be tracked over time.

## Running the Demo
A top-to-bottom live demo is provided which synthesizes realistic data, trains a real ML model, performs the quality check, and demonstrates safe payment failure handling.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full demo
python demo.py
```

## Running the Tests
We include automated tests for the critical payment failure handling logic (idempotency, mid-transaction failures, and reconciliation).

```bash
pytest tests/test_payment.py
```
