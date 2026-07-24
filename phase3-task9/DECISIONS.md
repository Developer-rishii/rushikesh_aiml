# Design Decisions — Task 9

Per the study guide (Section 8): pick one approach deliberately, and be
able to say why the other was rejected.

## 1. Interleaving vs. classic A/B for ranking evaluation
**Chosen: classic A/B (bucketed traffic split).**
- Interleaving (merging two ranked lists and attributing clicks to whichever
  model contributed the clicked item) needs far less traffic to detect small
  ranking-quality differences and is the academically preferred method for
  *pure relevance* comparisons.
- Rejected here because Task 9's bar is explicitly "ship to 10% of traffic
  and know within days" using **downstream business metrics** (applications,
  not just clicks) and a **permanent holdout for cumulative value** —
  interleaving doesn't support outcome metrics like conversion_rate or a
  stable long-run holdout the way bucketed A/B does. A/B is also far simpler
  to explain to a non-ML stakeholder who has to sign off on a hiring-relevant
  change, which matters given the fairness/DPDP constraints this system
  operates under.
- Trade-off accepted: A/B needs more traffic and more days to reach
  significance than interleaving would for the same effect size. Acceptable
  given PlaceMux's marketplace scale.

## 2. Permanent holdout vs. periodic switchback tests
**Chosen: permanent holdout (fixed 5% of users, carved out with a salt that
is never reused).**
- Switchback tests (toggle the whole system between variants on a schedule)
  are good for detecting network/marketplace effects (e.g. one side of the
  market affecting the other) and need less standing infrastructure.
- Rejected as the *primary* mechanism because Task 9 asks specifically for
  "cumulative model value" — a number that has to hold meaning across many
  successive experiments over months. A switchback resets its comparison
  window every cycle; a permanent holdout accumulates a stable "what if we
  had never shipped any ranking model" baseline that every future experiment
  can be measured against without re-deriving a control.
- Trade-off accepted: 5% of users permanently see a worse experience
  (baseline-only) for as long as PlaceMux keeps improving the model. This is
  the standard cost of a permanent holdout and is why it is kept small (5%,
  not 20%).

## 3. Pointwise regression scoring vs. pairwise/listwise learning-to-rank
**Chosen: pointwise regression (Ridge / Gradient Boosting) on relevance,
then rank by predicted score.**
- Pairwise/listwise objectives (LambdaMART, listwise cross-entropy) directly
  optimise ranking order and typically outperform pointwise approaches on
  nDCG in production ranking systems.
- Rejected for *this* deliverable only because Task 9's scope is the
  experimentation/guardrail layer, not the ranker itself — pointwise scoring
  is enough to produce two genuinely different, genuinely comparable model
  versions to exercise the experiment framework against. Swapping in a
  LambdaMART model later requires no change to `experiment_framework.py`,
  `guardrails.py`, or `fairness.py` — that decoupling is intentional.
- Flagged in `README.md` Section "Go deeper" as the next real improvement.

## 4. Synthetic cohort label instead of a real protected attribute
**Chosen:** the fairness audit uses a synthetic, non-identifying
`segment` label (A/B) instead of any real protected characteristic.
- Rejected using or simulating a real protected attribute (e.g. gender,
  age band) even synthetically, to avoid modelling patterns that could be
  mistaken for real demographic bias claims, and to keep the artifact clear
  of anything resembling real candidate PII per the DPDP note in the
  prerequisites. The parity-gap *mechanism* (rank-based selection rate,
  hard-guardrail threshold, halted on breach) is identical to what a real
  protected-attribute audit would use — only the label is a placeholder.
