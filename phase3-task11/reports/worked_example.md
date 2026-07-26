# Worked example (Stage B.4 / C.4 / D.4: "this input, this output, this plain-English reason")

**Job:** `job_id = 3` (held-out test job, 20 candidates shown)
**Candidate surfaced #1 by the chosen model:** `candidate_idx = 10`

## Input (the candidate's features, as logged)
| feature | value |
|---|---|
| skill_match | 0.562 |
| experience_match | 0.579 |
| embedding_sim | 0.845 |
| recency | 0.739 |
| past_response_rate | 0.456 |
| profile_completeness | 0.788 |

## Output
- **Model score: 6.240** → ranked **#1 of 20** by the chosen model
  (pairwise, IPS-corrected).
- The old heuristic ranked this same candidate **#2** — reasonable, but
  not #1.
- This candidate's **true relevance sits at the 90th percentile** among
  the 20 candidates for this job (a genuinely strong match — this is the
  simulator's hidden ground truth, used here only to verify the example,
  never available to the model).

## Plain-English reason
The model ranked this candidate #1 mainly because of a strong
`embedding_sim` (0.845, contributing +1.43 to the score) and a strong
`profile_completeness` (0.788, contributing +1.46) alongside a solid,
above-average `skill_match` (0.562, contributing +1.41) — the three
largest contributions to the total score. `past_response_rate` (0.456)
contributed the least (+0.14), reflecting the model's learned weight
that this signal, while genuinely predictive, is smaller in this
candidate's specific feature mix than the others.

## What happens if the model is unavailable
`serve.py`'s `Ranker.rank()` wraps scoring in a try/except: if the model
weights file is missing or a required feature is absent, it falls back
to `heuristic_baseline.score()` (itself hardened to skip missing
features and renormalize, rather than fail a second time), tags the
output `served_by = "heuristic_fallback"`, and logs a warning — the same
candidate would still be ranked and shown, just by the old heuristic
until the model issue is fixed. Verified in `tests/test_failure_and_bias.py`
(2 induced failure scenarios, both degrade safely — 5/5 tests passing).
