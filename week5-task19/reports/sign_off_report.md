# PlaceMux Task 19 Sign-Off Report

## Model Evaluation
The ML model was evaluated on a strictly held-out item-level test set, preventing label leakage. Items with fewer than 20 responses were excluded from metrics to ensure reliable ground truth representation.

### Baseline vs Model Performance

| Metric | Baseline | ML Model |
|---|---|---|
| Precision | 0.3333 | 0.8667 |
| Recall | 0.1176 | 0.7647 |
| FPR | 0.0784 | 0.0392 |
| AUC | N/A | 0.9746 |

