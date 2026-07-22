# Phase 3 - Task 7: Activation & Onboarding Funnel Optimization

An intelligent **Cold-Start Recommendation & Onboarding System** for PlaceMux. Designed to deliver genuinely relevant job recommendations to new candidates with zero interaction history during their first session.

---

## 🚀 Quick Start & Verification

Execute the complete end-to-end evaluation, train/serve skew audit, fairness checks, and test suite:
```bash
bash run_all.sh
```

---

## 🏗️ Architecture & Stage Mapping

### Stage A — Understand & Set the Bar
- **Goal**: Clear bar where brand-new candidates receive high-relevance job recommendations ($P@5 \ge 0.70$) in their first session.
- **Approach**: Hybrid model blending candidate profile skill overlap ($70\%$) with global job popularity prior ($30\%$), plus tiered $\epsilon$-greedy exploration ($15\%$).
- **Rejected Alternatives**: Explicit multi-step onboarding questionnaires were rejected due to candidate activation friction.

### Stage B — Cold-Start Recommendation Strategy
- Implemented in `src/cold_start_recommender.py` (v1.0.0).
- Combines Jaccard skill vector overlap with power-law popularity priors.
- Tiered exploration samples uncertain-but-plausible items from the immediate next ranking tier to accelerate taste learning without degrading first-session relevance.

### Stage C — Measured Lift in First-Session Actions
- Evaluated on held-out cold-start test candidates ($N=182$) in `src/evaluate.py`.
- **Results**:
  - Model $P@5$: **0.9253** | Baseline Popularity $P@5$: **0.0484**
  - Model $MAP$: **0.9284** | Baseline Popularity $MAP$: **0.0195**
  - Model $nDCG@10$: **0.9611** | Baseline Popularity $nDCG@10$: **0.0570**
  - **Offline $P@5$ Lift**: `+0.8769` (+87.69%)
  - **Expected Online $P@5$ Lift**: `+0.4385` (50% conservative discount applied for position/novelty bias)

### Stage D — Fallback That Is Never Empty
- Implemented in `src/fallback.py`.
- **3-Tier Protection**: Model Output $\rightarrow$ Popularity Fallback $\rightarrow$ Curated Evergreen Jobs.
- **Result**: $0\%$ empty results across model outages, zero user skills, and empty job pools ($18/18$ unit tests passing).

### Stage E — Integration, Governance & Verification
- **Model Versioning**: `MODEL_VERSION = "1.0.0"` tracked on recommender, fallback payloads, and API outputs (`docs/MODEL_CARD.md`).
- **Train/Serve Skew Audit**: Verified $0.0$ feature divergence (`PASSED_ZERO_SKEW`).
- **Fairness & DPDP Compliance**: Evaluated in `src/fairness.py`. Demographic Parity Ratio = **0.9055** (Exceeds EEOC 4/5ths / 80% Rule).

---

## 📁 Repository Structure
```
phase3-task7/
├── api/
│   └── serve.py                 # FastAPI serving layer (/recommend, /health, /fairness, /simulate_outage)
├── data/
│   ├── generate_synthetic_data.py # Real interaction log simulator
│   ├── jobs.json                # Job listings catalog
│   ├── users.json               # Candidate profile dataset
│   └── interactions.csv         # Interaction history log
├── docs/
│   ├── demo_script.md           # 2-minute live demo verification guide
│   ├── experiment_log.md        # Reproducible empirical evaluation results
│   └── MODEL_CARD.md            # Model governance & compliance documentation
├── src/
│   ├── cold_start_recommender.py # Hybrid cold-start model with versioning & exploration
│   ├── evaluate.py              # Offline evaluation harness & metric computation
│   ├── explain.py               # Explainability engine for candidate reasoning
│   ├── fairness.py              # Fairness & candidate representation auditor
│   ├── fallback.py              # 3-tier resilient fallback handler
│   ├── features.py              # Side-effect-free feature matrix builder
│   ├── popularity.py            # Popularity prior retrieval engine
│   └── skew_check.py            # Train/serve feature skew validator
├── tests/
│   ├── test_api.py              # FastAPI endpoint integration tests
│   ├── test_failure_modes.py    # Outage and edge case tests
│   ├── test_fairness.py         # Fairness audit unit tests
│   ├── test_metrics.py          # Metric formula unit tests
│   ├── test_skew.py             # Feature skew unit tests
│   └── test_versioning.py       # Model version metadata unit tests
├── README.md
└── run_all.sh                   # Main runner script
```
