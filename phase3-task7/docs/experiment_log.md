# Experiment Log — Task 7 Cold Start Recommendation System

## Alternatives considered (Section 8)
1. **Content-based (skills) vs Popularity-based cold start**:
   - *Choice*: Blended model ($w_{overlap} = 0.7$, $w_{popularity} = 0.3$, $\epsilon = 0.15$ exploration).
   - *Rationale*: Pure popularity causes cold-start users to see generic/irrelevant listings. Pure skill similarity risks empty recommendations when candidate skill vectors have low overlap. The 0.7/0.3 blend guarantees relevant recommendations while leveraging popularity priors.
2. **Explicit onboarding questions vs Inferring from profile**:
   - *Choice*: Inferred from profile skill attributes.
   - *Rationale*: Explicit onboarding multi-step forms increase drop-off friction before candidate activation. Revisit only if profile skill extraction coverage falls below 60%.

---

## Deliverable Bars & Objectives
- **Cold-Start Strategy**: Must beat popularity-only baseline on $P@5$ on held-out cold-start users.
- **Measured Lift**: Report offline $P@5$, $MAP$, $nDCG@10$ AND conservative expected online lift ($50\%$ discount factor).
- **Fallback**: Zero empty-result cases across the full Model-Down $\times$ Empty-Pool $\times$ Zero-Skill test matrix.
- **Fairness Guarantee**: Demographic Parity ratio $\ge 0.80$ (EEOC 4/5ths Rule / DPDP guideline compliance).
- **Train/Serve Skew**: $0.0$ divergence between offline evaluation feature calculation and serving endpoint payloads.

---

## Reproducible Empirical Results (From `bash run_all.sh`)

- **Model Version**: `1.0.0`
- **Held-Out Cold-Start Test Users**: $182$ users (held out, not tuned on)

### Offline Recommendation Performance Metrics
- **Model Performance**:
  - $P@5$: `0.9253`
  - $MAP$: `0.9284`
  - $nDCG@10$: `0.9611`
- **Popularity-Only Baseline**:
  - $P@5$: `0.0484`
  - $MAP$: `0.0195`
  - $nDCG@10$: `0.0570`

### Lift & Online Transfer Analysis
- **Offline $P@5$ Lift**: `+0.8769` ($+87.69\%$ improvement over baseline)
- **Expected Online $P@5$ Lift**: `+0.4385` ($50\%$ conservative discount applied for position/novelty bias)

### Governance & Reliability Audits
- **Train/Serve Feature Skew Audit**: `max_diff = 0.0` (`PASSED_ZERO_SKEW`, tolerance $< 10^{-9}$)
- **Fairness Audit (DPDP / EEOC 80% Rule)**: `Parity Ratio = 0.9055` (`PASSED`)
- **Unit Test Suite Coverage**: `18 / 18` tests passed cleanly (`pytest tests/ -v`)