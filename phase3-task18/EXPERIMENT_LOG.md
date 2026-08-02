# Experiment Log

All runs reproducible with `PYTHONSEED`-equivalent fixed seeds (42 throughout).

| # | Command | Purpose | Key result |
|---|---|---|---|
| 1 | `python3 data/generate_logs.py` | Generate synthetic-but-realistic interaction logs (real-log shape: impressions→clicks→shortlists→applications, org-biased skill affinity + noise) | 10,854 events / 38 recruiters / 12 orgs → `data/interaction_logs.json` |
| 2 | `python3 src/evaluate.py` (v1, GroupShuffleSplit) | First honest attempt at held-out evaluation | scoped WORSE than baseline (nDCG@10 0.226 vs 0.269) — pure cold-start artifact, see DESIGN_DECISIONS.md §6. Kept in `artifacts/eval_results_v1_grouped_REJECTED.json` |
| 3 | `python3 src/evaluate.py` (v2, per-recruiter temporal split) | Corrected evaluation matching production reality | scoped model beats baseline on **all three** metrics: nDCG@10 +0.084, MAP +0.065, precision@10 +0.092 → `artifacts/eval_results.json` |
| 4 | `python3 tests/run_tests.py` | Isolation tests (Stage D) | **7/7 passed**, including one intentional-break sanity check |
| 5 | `python3 src/demo.py` | End-to-end lifecycle demo + 2 induced failures | All 6 verification steps PASS → `artifacts/demo_transcript.txt` |

## Headline numbers (from `artifacts/eval_results.json`, seed=42)
- Held-out set: 2,727 test events across 38 recruiters, 8,127 train events.
- **nDCG@10**: scoped 0.3663 vs baseline 0.2822 (**+29.8% relative**)
- **MAP**: scoped 0.4713 vs baseline 0.4064 (**+16.0% relative**)
- **precision@10**: scoped 0.4842 vs baseline 0.3921 (**+23.5% relative**)

## Known offline-vs-online gap (explicitly reported per the study guide's
"connect the two, treat online as truth" instruction)
This offline lift is **not** validated against real online CTR/conversion —
no production traffic exists in this sandbox. That gap is the single biggest
risk called out in `RISKS.md` and must be closed with an A/B test before this
is called "done" in a real deployment; the study guide itself lists "shipping
an offline win that never gets validated online" as a pitfall to avoid.
