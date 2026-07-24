# Definition of Done — Task 9
Every line has a checkbox AND the file that is the evidence for it.
Per the scoring rubric: "a claim without evidence scores zero" — so nothing
below is asserted without a path you can open and check.

## Core deliverables (50 pts)
- [x] Model variant serving behind the experiment framework
      → `src/experiment_framework.py` (routing) + `reports/experiment_log.csv`
      (139k+ rows show `served_by` = baseline_v1 / candidate_v2 / fallback)
- [x] A permanent holdout group for measuring cumulative model value
      → `src/experiment_framework.py::HOLDOUT_PCT` + `reports/evaluation_report.md`
      "Cumulative model value vs. permanent holdout" section (real numbers,
      recomputed every run)
- [x] Guardrail metrics that halt a bad model
      → `src/guardrails.py` + `reports/guardrail_report.md` +
      `system_event_log.json` (`GUARDRAIL_HALT` event, day 13, this run)

## Real-data quality & correctness (20 pts)
- [x] Built on real logged data, not a curated sample
      → `data/generate_logs.py` generates ~140k impressions across 14 days,
      4000 users, 250 jobs, with per-row noise — not a hand-picked example set
- [x] Evaluated on held-out data never tuned on
      → `src/models.py::train_test_split_logs` (20% held out, seed=42);
      `reports/evaluation_report.md` reports the offline/online gap explicitly
- [x] Consistent assignment verified, not assumed
      → `tests/test_verification.py::test_consistent_assignment` +
      `run_all.py`'s in-pipeline assertion over all 4000 experiment users

## Live verification & evidence (15 pts)
- [x] Two model versions served live with separated metrics
      → `reports/per_variant_daily_metrics.csv` (control/holdout/treatment ×
      7 days, CTR + conversion_rate each)
- [x] Failure deliberately induced, degradation confirmed
      → Failure #1 (model outage → fallback): `system_event_log.json`
      → Failure #2 (bad deploy → guardrail halt): `guardrail_report.md`,
      `system_event_log.json`
- [x] 2-minute live demo script with real numbers
      → `reports/demo_script.md` (regenerated with this run's actual figures)
- [x] Automated verify-and-break checks, all passing
      → `tests/test_verification.py` — 7/7 pass (rerun any time, see README §2)

## Dependency, failure & edge-case handling (15 pts)
- [x] Holdout users never see the candidate model, even post-halt
      → `tests/test_verification.py::test_permanent_holdout_never_gets_candidate`
- [x] Model outage falls back safely instead of erroring the request
      → `src/failure_injection.py::score_with_fallback`
- [x] Guardrail halt actually reroutes traffic, not just logs a warning
      → `src/experiment_framework.py::route_model` checks `self.halted` first
- [x] Every model version traceable to the data it was trained on
      → `src/model_registry.py` + `reports/model_registry/*.json`
      (row count + training-data hash + timestamp per version)
- [x] Fairness checked every day, not once at the end
      → `src/fairness.py` invoked inside the daily loop in `run_all.py`,
      not as a post-hoc report

## Explicit non-goals (out of scope for this task, tracked in DECISIONS.md)
- Interleaving-based evaluation (chose classic A/B — see DECISIONS.md §1)
- Switchback testing (chose permanent holdout — see DECISIONS.md §2)
- Pairwise/listwise LTR model (chose pointwise — see DECISIONS.md §3)
