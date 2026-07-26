# Design decisions (Stage A.3: "write down WHY, including what you rejected")

## 1. What good looks like / the bar
Good = a ranker that beats `heuristic_baseline.py` on **nDCG@10** on jobs
it never saw in training, using only features that are actually
available at serving time (no position, no outcome data). The bar is
**beat the heuristic offline**; anything not beating it does not proceed
to an online test proposal.

## 2. GBDT/LambdaMART vs linear pairwise — REJECTED LambdaMART, not by choice
The natural choice for "pairwise/listwise LTR" is LightGBM's
`lambdarank` objective or XGBoost's `rank:pairwise`. **Neither is
installed in this sandbox, and the sandbox has no network access to
install them** (`pip install lightgbm` fails with no matching
distribution). This is a real infra constraint, documented rather than
hidden.

Rejected workaround: approximate a GBDT pairwise ranker by scoring each
candidate against a zero-reference vector. Rejected because that scoring
trick is only valid for models that are linear in the input — using it
with trees would silently give wrong scores at serving time, which is
worse than not having the fancy model at all.

**Chosen instead:** a linear pairwise ranker (RankSVM/RankNet-style
logistic regression over feature differences within each job). This is a
real, production-proven LTR family, and critically its score function
`score(x) = w · x` is exactly correct to apply to a single candidate at
serving time — no approximation needed.

**Stated production recommendation:** re-fit with LightGBM's `lambdarank`
objective on the exact same `features.py` / label pipeline the moment
infra allows it. Nothing else in this pipeline (data prep, position-bias
correction, evaluation, serving contract) needs to change — only
`train_ltr.py`'s model-fitting call.

## 3. Pairwise vs listwise (the guide's named "alternative approaches")
Both were built and evaluated on the identical, IPS-corrected labels and
identical held-out test jobs (see `reports/metrics.json`):

| Model | nDCG@10 | MAP@10 | Precision@5 |
|---|---|---|---|
| heuristic (current production) | 0.9137 | 0.7680 | 0.6344 |
| pairwise, corrected (**chosen**) | **0.9452** | **0.8353** | **0.7422** |
| listwise, corrected (considered) | 0.9316 | 0.8082 | 0.6978 |

**Chosen: pairwise.** It wins on every metric here. Plausible reason:
listwise ListNet-style training optimizes a single per-job softmax
distribution, which is more sensitive to how many candidates are IPS-
reweighted per job (variance) than pairwise's many independent local
comparisons — pairwise effectively gets more, smaller, more robust
training signals per job in this data regime (~20 candidates/job).
**Rejected, not discarded:** listwise still clearly beat the heuristic
(+2.0% nDCG@10), so it remains a documented fallback candidate if the
online test of the pairwise model underperforms its offline promise.

## 4. Raw vs IPS-corrected labels — the position-bias decision
See `reports/position_bias_ablation.md` for the full evidence. Summary:
correction is not optional here — the raw-label model's learned weight
for `recency` (a feature with **zero** true relevance in this data)
exceeds its weight for `skill_match` (the single most relevant feature),
because `recency` is exactly what the current heuristic over-ranks, so
recency-heavy candidates get examined and clicked more regardless of
true fit. IPS correction reduces (does not fully eliminate — reported
honestly) this confound and the corrected model's ranking of feature
importance matches the true relevance ordering far more closely.

## 5. What we did NOT do, on purpose
- No online A/B test — out of scope for this task per the study guide
  (Stage E hands off "candidate ranker for online experiment", it does
  not run one).
- No deep/neural embeddings beyond the given `embedding_sim` feature —
  out of scope; the task is about the ranking layer, not representation
  learning.
- No full counterfactual (SNIPS) policy-value estimate — flagged as a
  natural next step in "Go deeper" rather than built, to keep this
  deliverable finishable and verifiable end-to-end within scope.
