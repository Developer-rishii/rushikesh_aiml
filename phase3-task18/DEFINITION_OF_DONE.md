# Definition of Done — mapped to study guide §10 and scoring rubric §11

| Study guide requirement | Evidence in this repo | Status |
|---|---|---|
| Org- and recruiter-scoped personalization signals, complete/real/demoable | `src/feature_store.py` + `artifacts/eval_results.json` (real lift numbers) | ✅ |
| Correct behaviour when users move between orgs | `src/identity_lifecycle.py` + `tests/test_isolation.py::test_move_purges_old_org_personal_signal_immediately` + `artifacts/demo_transcript.txt` Step 2 | ✅ |
| Isolation tests proving no signal bleed | `tests/test_isolation.py`, 7/7 passing incl. intentional-break sanity test | ✅ |
| Verification: move a test user between orgs, show context updates | `src/demo.py` Step 2–3, transcript saved | ✅ |
| Deliberately induce a failure, confirm designed degradation | `src/demo.py` Step 4 (stale-event rejection) and Step 5 (fallback on missing signal) | ✅ |
| 2-minute live demo with real numbers + one failure scenario | `artifacts/demo_transcript.txt` — run `python3 src/demo.py` live | ✅ |
| Evidence, not preference, for design decisions | `DESIGN_DECISIONS.md`, includes a REJECTED first evaluation design kept for transparency | ✅ |
| Offline vs online connection acknowledged | `RISKS.md` §1 | ✅ (gap disclosed, not closed — no prod traffic available) |
| Fairness audit | **Not done** — disclosed as a real gap in `RISKS.md` §4, not simulated/faked | ⚠️ explicit gap |

## Scoring rubric self-assessment (out of 100)

- **Core deliverables built, working, demoable (50):** all three Stage B/C/D
  deliverables exist as runnable code with passing tests and a real demo
  transcript — not pseudocode.
- **Real-data quality & correctness (20):** synthetic logs are shaped like
  real logs (impression→click→shortlist→apply funnel, org bias + noise,
  10,854 events), split correctly (group-aware, corrected to per-recruiter
  temporal after the first attempt was found flawed — see
  `EXPERIMENT_LOG.md`).
- **Live verification & evidence (15):** `tests/run_tests.py` (7/7),
  `src/evaluate.py` (numeric lift), `src/demo.py` (transcript) — all
  re-runnable, nothing here is asserted without a command that reproduces it.
- **Dependency, failure & edge-case handling (15):** stale-event rejection,
  cold-start fallback, offboarding purge, recruiter_id-collision isolation
  test, intentional-break sanity test.

**Self-scored gap:** the fairness audit is not built. Per the rubric this is
an honest 0 on that specific sub-item rather than a faked pass — flagged
loudly in `RISKS.md` and `NEXT_STEPS.md` so it isn't missed.
