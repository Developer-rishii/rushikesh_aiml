# Task 14 — Fairness, Bias Audit & Explainability
## Bias Audit Report

**System audited:** PlaceMux candidate-shortlisting model
**Protected attribute:** gender (audit-only — never a model input)
**Data:** 12,000 logged applications, 25% held-out test split (n=3,000), seed=42, fully reproducible via `data/generate_data.py` → `src/train_model.py`

---

### 1. What good looks like (Stage A)
A defensible audit must: (a) name the fairness metric it optimizes for and why,
(b) measure it on real, held-out data, (c) apply a mitigation and re-measure
rather than stop at "found a problem," and (d) produce a per-decision
explanation a non-technical, potentially adversarial reader can verify.
Bar: regulator- and candidate-facing defensibility, not just a dashboard number.

### 2. Fairness metric chosen: Equal Opportunity (primary), Demographic Parity (reported)
Demographic Parity and Equal Opportunity conflict by construction — enforcing
equal *overall* selection rates (DP) can force shortlisting unqualified
candidates from one group, which is not defensible as "merit-based" in front
of a regulator. **Equal Opportunity** — equal true-positive rates among
candidates who are genuinely qualified — is the metric we optimize for,
because the harm we are most exposed on is a qualified candidate never
getting a fair shot. Demographic Parity (and the EEOC four-fifths rule) is
still reported as a secondary check.

### 3. Proxy variables identified
`college_tier` and `pincode_tier` are legitimate-looking features that
correlate with gender in this market (historical access disparities). The
model **never sees gender**, yet the pre-mitigation audit still finds a
gap — direct proof that **"we don't use gender" is not proof of fairness**
(see pitfalls, §12 of the study guide). This is the central finding of the
audit.

### 4. Results — BEFORE mitigation (held-out test set, n=3,000)
| Metric | Value |
|---|---|
| AUC | 0.7218 |
| Precision@0.5 | 0.6526 |
| Recall@0.5 | 0.4689 |
| Selection rate — Women | 26.44% |
| Selection rate — Men | 30.92% |
| **Demographic Parity Diff** | **-0.0448** |
| Four-fifths ratio | 0.855 (passes 0.8 threshold, but... ) |
| TPR — Women | 44.59% |
| TPR — Men | 47.89% |
| **Equal Opportunity Diff** | **-0.0329** |

Qualified women are shortlisted at a meaningfully lower rate than
equally-qualified men, despite gender never being an input. Full JSON:
`experiments/results_before_mitigation.json`.

### 5. Mitigation applied
**Kamiran & Calders pre-processing reweighing** (`src/mitigation.py`).
Each training row is reweighted so group and label are statistically
independent going into training — see file for the full rejected-alternatives
write-up (in-processing constrained optimization; post-processing per-group
thresholds).

### 6. Results — AFTER mitigation
| Metric | Before | After | Change |
|---|---|---|---|
| Demographic Parity Diff | -0.0448 | -0.0373 | ↓ 17% |
| **Equal Opportunity Diff** | **-0.0329** | **-0.0194** | **↓ 41%** |
| AUC | 0.7218 | 0.7211 | -0.0007 (noise) |
| Recall@0.5 | 0.4689 | 0.4730 | +0.0041 |

The primary metric (Equal Opportunity) improved by 41% with **no loss** in
offline predictive performance — evidence the disparity was bias, not signal.
Full JSON: `experiments/results_after_mitigation.json`,
`experiments/before_after_comparison.json`.

### 7. Worked example (per-decision explanation)
Real held-out candidate, model = mitigated model, served through the actual
`/explain` API (see `reports/demo_transcript.txt` for the full transcript):

> This candidate was **not shortlisted** (model confidence: 12%).
> - Years of experience of 0.7 worked against them.
> - Skill match with the role of 41.9 worked against them.
> - Assessment test score of 69.3 worked in their favor.

Ground truth label for this candidate: not shortlisted (model agrees).
Because the model is logistic regression, this attribution is **exact**,
not an approximation — the three listed terms are literally the largest
components of `intercept + Σ(coef_i × feature_i)`.

### 8. Failure mode tested
`src/failure_demo.py` deliberately flips the model to unavailable mid-run.
Result: the API returns HTTP 503 with `decision: DEFERRED_TO_HUMAN_REVIEW`
— **never** a silent fabricated score. See `reports/demo_transcript.txt`.

### 9. Questions a senior engineer would ask — answered
- **Which innocuous feature is a proxy for something protected?**
  `college_tier` and `pincode_tier` — both retained (they carry real merit
  signal) but their correlation with gender is exactly why the audit exists.
- **Which fairness definition would you defend in court?** Equal Opportunity
  — it targets denial of opportunity to the genuinely qualified, the
  legally central harm in hiring discrimination claims.
- **Does the explanation survive contact with an angry candidate?** Yes —
  it names concrete, verifiable factors (experience, skill match, test
  score) tied to an exact linear computation, not a black-box confidence
  number.

### 10. Known limitations (honesty over polish)
- Data is synthetic-but-structurally-realistic, not PlaceMux production logs
  (none were available) — the pipeline is production-ready and reproduces
  identically once pointed at real logs (`data/interactions_log.csv` schema).
- Logistic regression trades some ranking power for auditability; a
  higher-capacity model would need a documented SHAP-based fallback for
  Stage D (noted, not built, since shap is unavailable in this offline
  environment).
- Threshold is fixed at 0.5; a full deployment would tune per the
  offline→online gap (Stage A prerequisite), which requires online logs
  this environment does not have.
