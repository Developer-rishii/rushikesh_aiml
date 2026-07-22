# Model Card: PlaceMux Cold-Start Recommendation System (v1.0.0)

## Model Overview
- **Model Name**: PlaceMux ColdStartRecommender
- **Model Version**: `1.0.0`
- **Release Date**: 2026-07-22
- **Model Type**: Hybrid Content-Popularity Scoring with Tiered Epsilon-Greedy Exploration
- **Primary Maintainer**: PlaceMux AI/ML Engineering Team

---

## Intended Use & Target Task
- **Target Task**: First-session job recommendation for new candidates with zero historical interactions (clicks, applies, shortlists).
- **Core Bar**: A brand-new candidate sees genuinely relevant jobs in their first session ($P@5 \ge 0.70$).
- **Out of Scope**: Long-term collaborative filtering for mature users with extensive interaction logs (handled by downstream recommendation models).

---

## Technical Architecture & Feature Engineering
- **Feature Matrix**:
  1. `skill_overlap`: Jaccard similarity between candidate profile skills and job requirement skills.
  2. `popularity`: Power-law prior normalized across current job catalog.
- **Scoring Function**:
  $$\text{Score}(u, j) = w_{\text{overlap}} \cdot \text{Overlap}(u, j) + w_{\text{popularity}} \cdot \text{PopNorm}(j)$$
  where $w_{\text{overlap}} = 0.7$ and $w_{\text{popularity}} = 0.3$.
- **Exploration Strategy**: Tiered Epsilon-Greedy ($\epsilon = 0.15$). Top $(1 - \epsilon)$ slots allocated to highest exploitation scores; $\epsilon$ slots sampled from the immediate next tier (uncertain-but-plausible items) to discover candidate preferences without tanking first-session relevance.

---

## Verification & Metrics Summary
- **Offline Evaluation (Held-out $30\%$ test split, $N=182$)**:
  - $P@5$: `0.9253` (Baseline: `0.0484`)
  - $MAP$: `0.9284` (Baseline: `0.0195`)
  - $nDCG@10$: `0.9611` (Baseline: `0.0570`)
- **Offline-to-Online Transfer**:
  - Offline $P@5$ Lift: `+0.8769`
  - Expected Online $P@5$ Lift: `+0.4385` ($50\%$ conservative discount factor applied)
- **Train/Serve Skew Audit**:
  - Maximum Feature Difference: `0.0` (Guaranteed zero skew by shared feature extraction functions)

---

## Governance, Fairness & DPDP Compliance
- **Fairness Metric**: Demographic Parity & Representation Parity across candidate skill tracks.
- **Demographic Parity Ratio**: `0.9055` (Exceeds EEOC 4/5ths / 80% Rule threshold of `0.80`).
- **Data Protection**: Zero storage of sensitive personal identity attributes in recommendation features.

---

## Fallback & Degradation Strategy
1. **Tier 1 (Model Operational)**: Hybrid personalized scoring with exploration tags.
2. **Tier 2 (Model Down / Outage)**: Popularity-based fallback (`source: "popularity_fallback"`).
3. **Tier 3 (Empty Job Catalog / Zero Pool)**: Curated evergreen listings (`source: "evergreen_fallback"`).
- **Guarantee**: $100\%$ non-empty guarantee across all execution conditions ($0\%$ empty screens).
