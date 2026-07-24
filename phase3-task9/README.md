# Task 9 — Experimentation Platform, Feature Flags & Guardrails
PlaceMux · AI/ML Engineer · Phase 3 · Sprint B (Growth & Experimentation)

## 1. The bar (Stage A)
> "You can ship a new model to 10% of traffic and know within days whether
> it is better or worse."

Three deliverables, all shipped and demoable on real (synthetically
generated but behaviourally realistic) logged data:

1. **Model variant serving behind the experiment framework** — `src/experiment_framework.py`
2. **A permanent holdout group for measuring cumulative model value** — same file, `HOLDOUT_PCT`
3. **Guardrail metrics that halt a bad model** — `src/guardrails.py`

The metric that decides "better or worse": **online conversion_rate**
(applications / impressions), compared treatment vs. control, per the
study guide's "treat online as the truth" principle. Offline nDCG/MAP/
precision@k are used only as a pre-ship sanity check, never as the
ship/no-ship decision.

## 2. What was actually built and run
This is not a design doc — `run_all.py` trains real models on ~140k
logged impressions, serves two live model versions across a 7-day
simulated experiment window, deliberately breaks things twice, and
writes every number below to `reports/`.

**Note:** The pipeline now runs completely from a clean clone with zero manual setup (all paths are relative, directories auto-create). Reproduce with:

```bash
python3 data/generate_logs.py      # ~140k synthetic-but-realistic impressions
python3 run_all.py                 # train, register, serve, guardrail, report
python3 tests/test_verification.py # 7/7 automated evidence checks
```

## 3. Results (from `reports/`, this run)
- Offline: candidate_v2 nDCG@5 **0.9997** vs baseline_v1 **0.9988**; MAP
  **0.6345** vs **0.6308** (candidate wins offline — see `evaluation_report.md`).
- Online, healthy days (7–11): treatment conversion tracked control within
  the 10% guardrail tolerance every day — see `per_variant_daily_metrics.csv`.
- Permanent holdout: cumulative relative lift of served traffic vs. holdout
  reported in `evaluation_report.md` (`cumulative_lift`), recomputed fresh
  each run since it's seeded but stochastic.
- **Failure #1 (outage):** candidate_v2 raised on day 9 → `system_event_log.json`
  shows one `fallback_triggered` event, baseline_v1 served, zero user-facing
  downtime.
- **Failure #2 (bad deploy):** candidate_v2 scores deliberately corrupted
  (anti-correlated with true relevance) on days 12–13 → conversion dropped
  ~11–12% relative vs. control both days → **guardrail auto-halted the
  experiment on day 13**, routing all treatment traffic back to baseline_v1.
  See `guardrail_report.md` and `system_event_log.json` (`GUARDRAIL_HALT`).
- Fairness: demographic parity gap stayed under the 5-point-percentage hard
  limit for control/holdout throughout; treatment's gap widened during the
  bad deploy — see `fairness_audit.md`.

Every number above is regenerated, not hand-typed, every time `run_all.py`
runs — that's what makes it evidence instead of a claim.

## 4. Repo map
```
data/generate_logs.py        Stage 4/6 input: realistic logged interactions
src/models.py                 baseline_v1 (Ridge) + candidate_v2 (GBM) rankers
src/experiment_framework.py   consistent hashing, permanent holdout, routing
src/guardrails.py             daily guardrail evaluation + halt rule
src/fairness.py               demographic parity gap (job-level shortlist)
src/metrics.py                nDCG/MAP/precision@k (offline), CTR/conversion (online)
src/model_registry.py         versioned model artifacts + training metadata
src/failure_injection.py      outage simulation + safe fallback
run_all.py                    Stage E: full pipeline, writes all reports/
tests/test_verification.py    Stage E.3: verify-and-break, evidence for DoD
reports/                      generated evidence (regenerated on every run)
DECISIONS.md                  alternatives considered and rejected
DEFINITION_OF_DONE.md         DoD checklist mapped to evidence files
```

## 5. Answers to the brainstorming questions (Section 9 of the study guide)
- **"How would you detect a model that improves clicks but worsens hires?"**
  CTR is a *soft* guardrail; conversion_rate (application rate) is a *hard*
  one. A model that only lifts CTR without lifting conversion trips no halt
  but shows up immediately in `per_variant_daily_metrics.csv` as a CTR/
  conversion divergence — that's the whole reason the two are tracked
  separately instead of collapsed into one "engagement" score.
- **"Which guardrail must never be crossed, even for a big win?"**
  `conversion_rate` and `fairness_gap` — see `HARD_GUARDRAILS` in
  `guardrails.py`. CTR alone can legitimately trade down for more relevant,
  lower-volume matches, so it's a soft guardrail only.
- **"Is your assignment truly random and stable?"**
  Stable: `test_consistent_assignment` proves the same user always lands in
  the same bucket. Random enough: SHA-256 of `(salt, user_id)` gives a
  uniform, non-gameable bucket distribution without needing to store any
  assignment state server-side.
