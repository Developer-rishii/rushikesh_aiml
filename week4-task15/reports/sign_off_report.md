# AI Trust Sign-Off: Proctoring False-Positive Reduction

## 1. What "good" looks like
False positives reduced vs baseline. The goal was to demonstrably lower the FPR from the v0 baseline on a held-out test split, without completely destroying recall.

## 2. Upstream dependency
Flagged-session data arrived from Task 11 with schema: `session_id`, `student_id`, `assessment_id`, `tab_switch_count`, `face_count_anomalies`, `copy_paste_events`, `time_per_question_zscore`, `network_latency_flag`, `webcam_dropout_seconds`, `flagged_by_v0`, `ground_truth_label`.
We enforce validation at load time. **Test that proves it:** `test_upstream_dependency_validation` confirms the pipeline fails loudly on malformed or empty batches.

## 3. Baseline (v0 rule system)
* Precision: 0.3810
* Recall: 1.0000
* False-Positive Rate (FPR): 1.0000
* Absolute False Positives: 13

## 4. Trained model
* **Logic & Features:** Binary Random Forest classifier using original signals plus derived features: `signal_combo_score` and `network_issue_derived` (high latency + webcam dropout). 
* **Threshold Selection:** Tuned on validation split. Chosen threshold: **0.10**. This threshold maximizes precision/FP reduction while forcing recall to stay >= 0.80.
* **Artifact Location:** `/src/models/proctor_model.pkl`

## 5. Baseline vs model comparison (THE Verdict)
✅ **FPR reduced from 1.0000 to 0.0000** 
* **Recall at chosen threshold:** 1.0000
* **Absolute FP count:** Went from 13 down to 0

### Segment Breakdown by FP Pattern:
* fp_network: Baseline flagged 6 -> Model cleared 6/6
* fp_copypaste: Baseline flagged 7 -> Model cleared 7/7
* (Numbers reflect the specific test-set slice evaluated)

## 6. Proof-of-detection
**Cleared Known FP Session:** `60a140c4-22e4-4cf0-8ab0-d9e65a6ff4df`
* v0 Flag: 1
* Model Verdict: cleared (Confidence: 1.00)
* Reason: Cleared: high tab-switch count (5.0) but network_issue_derived=1 — consistent with known connectivity FP pattern; model confidence 1.00 that this is NOT a violation.

**Confirmed Known TP Session:** `97ee4330-6aa6-4be8-b577-55148fbd1acf`
* v0 Flag: 1
* Model Verdict: flagged (Confidence: 1.00)
* Reason: Flagged: Model confirms violation driven by tab_switch=8.0, face_anomalies=4.0, copy_paste=8.0. Confidence 1.00.

## 7. Edge cases tested
* **Sensor fault:** Tested via `test_sensor_fault_routing` (routes to `no_data`).
* **Duplicate:** Tested via `test_duplicate_session_handling` (deduped by `keep='last'`).
* **Threshold edge:** Tested via `test_threshold_edge_case` (deterministic `>=` handling).
* **Upstream validation:** Tested via `test_upstream_dependency_validation`.

## 8. Out of Scope Notice
Verification/interview-scheduling and e-sign tamper-evidence are out of scope for this AI/ML proctoring slice; those are owned by the Integration/Full-Stack team.

## 9. Hand-off: Trustworthy proctoring
* **Next Team Schema:** Expect JSON with `verdict` (flagged/cleared/no_data), `confidence` (float), `reason` (string), and `fp_pattern`. 
* **Threshold:** Use 0.10 in production.
* **Guardrail:** Re-evaluate FPR monthly as more sessions get reviewed labels; alert if FPR creeps back above 0.10.
