# Leakage Check Report

## Domain-reasoned leaks (removed by definition, not just correlation)

- **pathologist_diagnosis_code**: Populated by the pathology lab only AFTER the diagnosis is finalized -- this is the target itself restated in another vocabulary (BENIGN-CONFIRMED / MALIGNANT-CONFIRMED), not a predictor available at the time of biopsy analysis.

## ID-like / no-signal features removed

- **patient_record_id**: Unique identifier, carries no generalizable signal; drop.

## Statistical smell test (|corr| > 0.9 with target)

- No numeric features exceeded the 0.9 threshold (note: `pathologist_diagnosis_code` is categorical, so it is caught by domain reasoning, not this numeric check -- proof that correlation-only leakage detection is not enough).

## Final dropped feature set: ['pathologist_diagnosis_code', 'patient_record_id']

## Remaining feature count after cleaning: 30 (the original 30 real WDBC measurements, excl. target)