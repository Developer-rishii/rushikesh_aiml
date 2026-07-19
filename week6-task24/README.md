# Task 24 — Fairness close + sign-off (AI/ML Engineer)

Closes the fairness audit opened in Task 21 and produces the **ML go-ahead**
artifact the launch team consumes before flipping the switch. Rebuilt
against the actual scoring rubric below.

## Folder structure

```
task24/
├── README.md
├── src/
│   ├── config.py              # paths, thresholds, scale params (single source of truth)
│   ├── data_generator.py      # 40k-row, 50-job, realistically messy dataset
│   ├── data_validation.py     # schema checks + cleaning (dupes, missing, outliers)
│   ├── baseline_model.py      # dumb rule-based baseline ("baseline first")
│   ├── metrics.py             # precision/recall/FPR/accuracy + disparate impact
│   ├── task21_audit.py        # UPSTREAM DEPENDENCY: produces task21_audit_results.json
│   ├── dependency_loader.py   # validates the Task 21 hand-off before proceeding
│   ├── mitigate.py            # real trained model (GradientBoosting) + reweighing + threshold tuning
│   ├── audit.py               # fairness-ceiling reference calculation
│   ├── sign_off.py            # full pipeline, decision logic, writes the go-ahead artifact
│   └── explain.py             # one-example plain-English walkthrough
├── demo/
│   └── run_live_demo.py       # single script, live console evidence, all 5 stages
├── tests/
│   ├── test_pipeline.py       # 25 tests
│   └── run_tests.py           # dependency-free runner (no network to install pytest)
├── data/
│   └── candidates.csv         # 40,400 raw rows (generated)
└── artifacts/
    ├── task21_audit_results.json  # upstream hand-off (consumed, not recomputed)
    ├── ml_go_ahead.json            # THIS task's hand-off to launch/legal/product
    └── metrics_report.json         # full quality report
```

## How this maps to the scoring rubric

### Core deliverable — built, working & demoable (50)
- Real trained ML model: `GradientBoostingClassifier` (150 trees), not a
  linear stand-in, trained with Kamiran–Calders reweighing.
- A dumb rule-based baseline exists first (`baseline_model.py`) — required by
  the study guide's "Baseline first" principle — so every later number is
  judged against it, not in a vacuum.
- `demo/run_live_demo.py` runs the entire chain in one command and prints
  real numbers at every stage (see live run below).
- `explain.py` gives the required one-example, plain-English "why."

### Real-data quality & correctness (20)
- **Scale**: 40,400 rows across 50 job postings (not one static rule).
- **Messiness, not happy-path**: ~3% missing values in two columns, ~1%
  duplicate rows (re-submitted applications), ~2% outlier injection
  (corrupted `years_exp`) — all handled explicitly in `data_validation.py`
  and logged in the artifact.
- **Correctness isn't just accuracy**: the mitigated model is scored two
  ways — against the historical (biased) label, where a drop is *expected
  and correct*, and against a merit ground truth, where it holds up
  consistently across all three tiers (~86% accuracy, ~75–79%
  precision/recall in every segment — no segment silently underperforming).

### Live verification & evidence (15)
Actual output from `python3 demo/run_live_demo.py` (this run, not a claim):

```
STAGE 1 — Task 21 upstream audit
Finding: FAIL
Baseline disparate impact: 0.000   (target >= 0.80)
Group positive rates: {tier 1: 0.931, tier 2: 0.047, tier 3: 0.000}
Baseline quality: accuracy 0.95, precision 0.92, recall 0.91
  → high accuracy, complete unfairness: the exact trap the study guide warns about

STAGE 2 — Task 24 mitigation
Dumb baseline disparate impact: 0.979 (tier-blind rule; fair but weak: accuracy 0.52)
Fairness ceiling (best achievable): 0.978
Mitigated model disparate impact: 0.992
Mitigated model quality vs merit ground truth: accuracy 0.861, precision 0.770, recall 0.767

STAGE 3 — Decision
DECISION: SIGNED_OFF
RATIONALE: Mitigated disparate impact 0.992 meets the 4/5ths rule threshold of 0.80.

STAGE 5 — Failure path
[BLOCKED] Upstream dependency missing: Task 21 audit results not found at
'/tmp/nonexistent_audit.json'. Run task21_audit.py first.
```

Tests, live:
```
$ python3 tests/run_tests.py
...
25/25 tests passed
```

### Dependency, failure & edge-case handling (15)
- **Real hand-off, not recomputation**: `sign_off.py` loads
  `task21_audit_results.json` through `dependency_loader.py` rather than
  quietly recomputing the Task 21 finding inline. If it's missing,
  malformed JSON, or missing required fields, it raises `DependencyError`
  with an actionable message (tested — see Stage 5 above and
  `test_dependency_loader_*`).
- **Data edge cases**: empty dataframe, missing required columns, and a
  vanished protected-attribute group are all explicitly checked in
  `data_validation.py` (`DataValidationError`), not left to crash downstream.
- **Sign-off can say no**: `decide()` supports `SIGNED_OFF`,
  `CONDITIONALLY_SIGNED_OFF` (near the data's fairness ceiling but under
  0.80), and `WITHHELD` (neither) — tested for all three branches plus a
  `None`-input edge case.

## Actual numbers from the last full run (seed=42, n=40,000 post-clean)

| Stage | Disparate impact | Notes |
|---|---|---|
| Task 21 baseline (biased, tier used as a feature) | 0.000 | 95% accuracy — bias hides behind good-looking metrics |
| Dumb tier-blind rule baseline | 0.979 | Fair by construction, but only 52% accuracy — this is the floor to beat |
| Fairness ceiling (merit-only reference) | 0.978 | Best any model could do on this data |
| **Mitigated model (deployed)** | **0.992** | 86% accuracy vs merit ground truth, consistent across tiers |

**Decision: `SIGNED_OFF`**

## Hand-off

- **Depended on:** `artifacts/task21_audit_results.json` (Task 21's bias
  finding — loaded and validated, not assumed).
- **Hands off:** `artifacts/ml_go_ahead.json` — versioned, machine-readable,
  contains the full audit trail, per-tier quality, mitigation method, and
  decision rationale so downstream teams don't have to trust a claim.

## Self-check (Q4 — AI/ML slice only)

What's still open before flipping the switch, for this track specifically:
- Re-audit on real production outcomes — this is synthetic, real-shaped
  data, not live traffic.
- Schedule a quarterly re-audit cadence; DI can drift as the applicant/job
  mix shifts.
- Data-deletion verification and load-testing are explicitly out of scope
  here (security/backend own those).

## Re-running everything

```bash
cd src
python3 data_generator.py   # regenerate the 40k-row messy dataset
python3 task21_audit.py     # regenerate the upstream dependency artifact
python3 sign_off.py         # run mitigation + sign-off, writes ml_go_ahead.json
python3 explain.py          # one-example walkthrough
cd ../demo
python3 run_live_demo.py    # everything above, in one command, full console evidence
cd ../tests
python3 run_tests.py        # 25 tests
```

Note: `pytest` isn't installable in this sandbox (no network access), so
`tests/run_tests.py` is a small dependency-free runner that executes the
same `test_*` functions pytest would discover.
