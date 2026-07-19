# Model Card — PlaceMux Candidate-Job Ranking Model

## Identity & versioning
- **Model version**: `model_v20260719_140957` (see `artifacts/models/LATEST_VERSION.txt` for the
  currently-serving pointer — this answers "which model produced a decision six months ago",
  flagged as a pitfall in the study guide: every scored decision logged in `predictions.jsonl`
  carries `model_version` and `feature_pipeline_version`).
- **Feature pipeline version**: `fp_v1.0.0` (`src/features/feature_pipeline.py`) — bumped whenever
  feature logic changes; serving refuses to start if a model's recorded pipeline version doesn't
  match the running code (see `ModelService._load` assertion).
- **Model class**: `sklearn.ensemble.GradientBoostingRegressor`, pointwise learning-to-rank
  (regress graded relevance 0-3, rank by predicted score).
- **Training data**: 60,000 impression rows, days 0-23 of `data/raw/interaction_logs.csv`
  (sha256[:16]=`d45ba6f76aa74a2e`, recorded in `metadata.json` for reproducibility).
- **Held-out test data**: 15,000 impression rows, days 24-29, never used in training or tuning.

## Intended use
Ranks jobs shown to a candidate in a search/recommendation results list. Feeds directly into
what candidates see first — a matching decision that materially affects a candidate's career,
which is why every prediction carries a plain-English explanation (see `docs/WORKED_EXAMPLE.md`)
and why quality is monitored continuously rather than validated once at launch.

## Objective & rejected alternative
Pointwise regression on graded relevance was chosen over full pairwise/listwise LambdaMART
because this sandbox has no network egress to install LightGBM/XGBoost. `build_model()` in
`src/training/train_model.py` isolates the model class behind one function specifically so
LambdaMART (or any other objective) can be swapped in later without touching feature code,
evaluation code, or serving code — the interfaces (`compute_features`, `.predict()`, `nDCG@k`
eval) don't depend on this choice.

## Performance (see `docs/EVAL_REPORT.md` for the full, timestamped report)
| Metric | Model | Baseline (popularity) |
|---|---|---|
| nDCG@10 (offline, held-out) | 0.8008 | 0.7342 |
| Precision@5 (offline, held-out) | 0.4000 | 0.3930 |
| Simulated online CTR (top-1 shown) | 0.5004 | 0.4040 |

## Feature importances (also used for per-prediction explanations at serve time)
skill_overlap_ratio (0.44) > exp_gap (0.33) > salary_gap_abs (0.15) > region_match (0.04) >
cand_activity_score (0.03) > job_popularity_log (0.01).

## Fairness
`src/fairness/fairness_check.py` compares the top-20%-score selection rate across `cand_region`
groups (demographic-parity-style check) as a *mechanism* demonstration — this dataset does not
encode any real protected attribute (gender, caste, religion, disability, etc. under India's DPDP
Act), since fabricating protected-class labels for a demo would be worse than not testing at all.
Result this run: max-min gap 0.0236 (threshold 0.10) → PASS. In production this same function
runs against real protected-attribute columns behind appropriate access controls, on every eval
run — not once at the end as a formality.

## Known limitations / future work
- Objective is pointwise, not listwise — a true LambdaMART/listwise loss would likely improve
  ranking quality further; blocked on package availability in this environment, not a design choice.
- Score-distribution monitoring, not full per-feature population-stability-index drift detection
  (documented as a deliberate scope choice in `docs/SLO_DEFINITION.md §6`).
- Synthetic data: relevance labels are generated from a known latent function (skill overlap +
  experience/salary/region fit) rather than real DPDP-governed candidate/job records, since real
  production logs aren't available in this environment. The pipeline (feature code, training code,
  monitoring code) is unchanged by swapping in real logs — only `src/data_generation/generate_logs.py`
  would be replaced by a real log export.
