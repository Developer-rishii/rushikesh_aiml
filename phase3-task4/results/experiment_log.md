# Experiment Log

All runs below were executed in this environment via `./run_all.sh` on
2026-07-19. Numbers will vary slightly run-to-run (real wall-clock load
test, real thread scheduling) but the shape — knee around concurrency
16–32, safe capacity ~280–295 QPS, fallback onset by concurrency 48–64 —
has been stable across repeated runs.

## Data generation
`python3 src/generate_data.py` → 24,000 rows, 400 jobs.
Funnel: click 14.1%, applied 4.0%, shortlisted 1.1% — a realistic sparse
marketplace engagement funnel, not a toy balanced dataset.

## Model training
`python3 src/train_model.py`
- Job-level `GroupShuffleSplit`: 300 train jobs / 100 held-out jobs
  (code asserts the job-id sets are disjoint — no leakage).
- `GradientBoostingRegressor`, 150 trees, depth 3, trained in ~3.7s on
  18,000 rows.
- Held-out metrics: nDCG@10 = 0.274, MAP@10 = 0.404, precision@10 = 0.262.
- Baseline (rank by `skill_match` alone): nDCG@10 = 0.242, MAP@10 = 0.398,
  precision@10 = 0.217.
- Gap (model − baseline): +0.033 nDCG@10, +0.006 MAP@10, +0.045 precision@10.
- Saved: `models/ranker_model.joblib`, `models/model_registry.json` (v1).

## Serving smoke test (cold start vs warm)
- Cold start (first request after boot): ~1980–2140ms.
- Warm request (same payload, subsequent calls): ~25–28ms.
- Cold start penalty: ~70–80x — this number drives the "pre-warmed minimum
  pool" decision in `docs/scaling_plan.md`.

## Load test (the numbers reports are built from)
Command:
```
python3 src/load_test.py --url http://127.0.0.1:8877/rank \
  --out results/load_test_results.csv --duration 3 \
  --levels 1,2,4,8,16,24,32,48,64,96,128
```
Result: `results/load_test_results.csv` (11 rows, one per concurrency level).
Latency stays flat (~30ms p95) up to concurrency 8; climbs sharply from
concurrency 16 onward; heuristic fallback starts engaging by concurrency
48 (~0.1%) and rises to ~34% by concurrency 128.
Error rate stayed at **0.0 at every single level up to 400 QPS attempted**
— the graceful-degradation design holds under the heaviest load tested.

## Failure injection
Command: `python3 src/failure_demo.py`
18 requests across 3 phases (healthy → forced-down → recovered).
Result: `results/failure_demo_log.json`, verdict `PASS: true`
(0 errors, fallback engaged 100% of the time during the outage, latency
dropped from ~28ms to ~2–3ms on the fallback path, recovered to
`used_fallback: false` immediately after `/admin/model_up`).

## Analysis + skew check
- `python3 src/analyze_results.py` → `results/latency_curve.png`,
  `results/breaking_point_report.md`.
  Breaking point: concurrency 32, ~290 QPS, p95 ≈ 193ms (~6.3x the
  concurrency-1 baseline of ~30ms).
- `python3 src/skew_check.py` → `results/skew_report.md`.
  Flagged 2/8 features (`exp_years`, `job_num_applicants_so_far`) as
  skewed between training and served traffic — root-caused to the
  load-test client using a fixed synthetic payload rather than sampled
  real traffic (see `docs/scaling_plan.md` §5 for the fix before sign-off).
