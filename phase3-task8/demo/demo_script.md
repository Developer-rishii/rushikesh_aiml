# 2-Minute Live Demo Script

**Setup (before the room):** `cd placemux_task08 && python run_pipeline.py` has
already been run once; have `outputs/` open in a second window.

## 0:00–0:20 — The bar
"We're not shipping a model, we're shipping an early-warning system. The bar
is: does it find at-risk candidates with enough lead time that growth can
still do something. Our label is 21 days of silence *ahead*, not detecting
someone who's already gone."

## 0:20–0:50 — Show the honest evaluation
Open `outputs/pr_curve.png`.
"On three months of held-out data the model never saw during training —
PR-AUC 0.61 vs 0.48 for the exact '14 days of silence' rule everyone already
uses. At the top decile we flag, precision is 0.73 vs the baseline's 0.47 —
we send growth a much cleaner list. I'll also say the uncomfortable part out
loud: at that same threshold our recall (0.31) is lower than the baseline's
(0.55) — the model is more precise but more conservative. That trade-off is
in `evaluation_report.md`, not hidden."

## 0:50–1:15 — Show one real decision
Open `outputs/worked_example.md`.
"Candidate C105970: risk score 0.93. Why: activity in the last 30/90 days is
well below typical for their tenure. They did in fact go silent for the next
21 days — this is a real holdout example, not a cherry-picked demo row."

## 1:15–1:35 — Show the at-risk list & the lever
Open `outputs/at_risk_list_for_growth.csv`.
"500 candidates, ranked, each with a suggested lever — 're-engagement push'
vs 'send a job digest' — because a risk score with no action attached is a
vanity model."

## 1:35–2:00 — Break it, live
Run: `python src/failure_simulation.py`
"Now I'm forcing the model artifact to fail to load — simulating the
scoring service being down." (point at the console output)
"See: `degraded_mode: true`, it falls back to the transparent 14-day rule,
and growth still gets a full list of the same size — the pipeline degrades,
it doesn't go silent. That assertion is checked in code, not just claimed:
`assert degraded_result["degraded_mode"].all()`."

## Anticipated questions (Section 9 of the study guide)
- *"Does it beat the 14-day rule?"* — Yes on PR-AUC/lift, no on recall at
  this threshold; both numbers shown, not just the favorable one.
- *"What's the actual lever?"* — Per-row `suggested_lever` column, driven by
  which specific behaviour (browsing-not-applying vs total silence) triggered
  the flag.
- *"Are you predicting churn or detecting it?"* — Predicting: already-dormant
  accounts (60+ days silent) are explicitly excluded from the label so we're
  never scoring the past.
