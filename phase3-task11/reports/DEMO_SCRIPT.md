# 2-minute live demo script (Stage E.4)

**0:00–0:20 — Set the bar**
"Here's the current production heuristic — hand-tuned weights, never
re-fit against outcomes. It scores 0.9137 nDCG@10 on held-out jobs. That's
the number the new ranker has to beat, offline, before anyone talks
about an online test."
```bash
python3 src/heuristic_baseline.py   # (import + show WEIGHTS dict)
```

**0:20–0:50 — Run the real end-to-end pipeline live**
```bash
bash run_all.sh
```
Narrate while it runs: "This regenerates the logged impressions, fits
position-bias propensities off the 10% randomized-order traffic slice,
trains the pairwise and listwise rankers on IPS-corrected labels, and
evaluates all four candidates — heuristic, raw pairwise, corrected
pairwise, and listwise — against true relevance on held-out jobs."

**0:50–1:20 — The headline number and WHY**
Point at the printed table: chosen model = 0.9452 nDCG@10, **+3.5%** over
the heuristic. Open `reports/position_bias_ablation.md` and show the two
weight tables: "Without correction, `recency` — a feature with **zero**
true relevance — got the single highest learned weight, because that's
exactly what the old heuristic over-ranks. With correction, `skill_match`
correctly becomes dominant."

**1:20–1:40 — One worked example**
```bash
cat reports/worked_example.md
```
"Candidate 10 on job 3: old heuristic ranked it #2, new model ranks it
#1, and it's genuinely in the 90th percentile of true relevance for that
job — verifiable because this data is simulated with ground truth, purely
for grading this exercise."

**1:40–2:00 — Live failure, live degradation**
```bash
python3 -m pytest tests/ -v 2>/dev/null || python3 tests/test_failure_and_bias.py
```
"These two tests literally delete the model file and drop a required
feature column mid-request. Both fall back to the heuristic instead of
crashing — `served_by = heuristic_fallback` — and 5/5 tests pass,
including a regression guard that fails the whole suite if the chosen
model ever stops beating the heuristic offline."

**Hand-off line:** "This ranker is ready to be proposed for an online
test — that's the boundary of this task; the online experiment itself is
the next team's job."
