# 2-Minute Live Demo Script — Task 20 Pilot Dry-Run

**Setup (before the room):** `python3 run_all.py` already run once; have
`experiments/`, `models/model_card.md`, `demo/worked_example.json` open.

**0:00–0:20 — Where this fits**
"This is the enterprise pilot dry-run for tenant AcmeFinServ_Pilot —
proving the ranking layer on realistic hiring data before we touch a
real customer's live pipeline."

**0:20–0:50 — Show the pilot run + honest evaluation**
Open `experiments/metrics_offline.json`. Say plainly: offline nDCG/MAP
did NOT clearly beat the recruiter's manual baseline, but the
real-outcome hire-capture proxy did. "We are not shipping this on the
offline number alone — that's exactly the offline/online trap the task
warns about, and it's why an online A/B test is remediation item #1."

**0:50–1:15 — Fairness**
Open `experiments/fairness_report.json`. "Demographic parity passes the
4/5ths rule, but we found an equal-opportunity gap between groups —
truly-qualified candidates in group A are surfaced less often. That's
flagged HIGH in the remediation list, not buried."

**1:15–1:35 — Explainability**
Open `demo/worked_example.json`. Read the one-line plain-English reason
out loud for candidate C00001 → job J003.

**1:35–1:55 — Break it live**
Run `python3 src/failure_test.py`. Point at the model-missing case:
"model file gone, and the tenant still gets a ranked list — labeled
baseline, not a 500 error."

**1:55–2:00 — Close**
"Full remediation list is in `docs/remediation_list.md`, seven items,
each tied to a number we actually measured, not a guess."
