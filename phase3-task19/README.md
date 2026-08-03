# Task 19 — White-Label Configurability & Admin Control Plane
PlaceMux · AI/ML Engineer · Sprint D (Enterprise Readiness)

## 1. The bar (from the study guide)
> An enterprise can tune matching to their hiring bar **without** being able
> to configure their way into a biased or broken system.

Three deliverables, all built, run on real (realistically-generated,
funnel-shaped, per-tenant) logged data, and demoed live:

1. **Configurable matching policy layer** (weights/rules) per tenant — `src/policy.py`
2. **Guardrails** against unfair/nonsensical configs — `src/guardrails.py`
3. **Admin preview** of a config's effect before it goes live — `src/preview.py`

## 2. Design decision & rejected alternative
**Chosen:** one tenant-agnostic base ranking model (`src/model.py`,
pointwise GradientBoosting) + a thin, bounded, versioned policy layer per
tenant on top (`src/policy.py`) that re-weights transparent sub-signals and
applies hard eligibility rules.

**Rejected: retraining a full model per tenant.** With 3 tenants today
and more coming (that's the point of "white-label"), per-tenant retraining
means N models to version, monitor for drift, and re-audit for fairness —
it doesn't scale operationally and makes the guardrail surface (Stage C)
much harder to reason about, since "unfair" would have to be checked
per-model instead of per-config. A shared model + bounded config is also
what makes **preview** (Stage D) cheap: previewing a reweight is a
matrix multiply; previewing a retrain is not something you can do live in
an admin console.

**Guardrail style chosen: hard rejection, not warnings.** The study guide's
bar is "can't configure their way into a biased system" — a dismissible
warning doesn't meet that bar; a blocked commit does. (See `demo.py` Step 4.)

## 3. Where train/serve skew is prevented
`src/features.py` is the **only** place feature values are computed. Both
`model.py` (training) and `serve.py` (serving) import `compute_features()`
from it — never duplicated inline. `assert_no_protected_attrs()` runs
before every training call so the protected-attribute proxy used to
simulate historical bias in the logs (`gender_proxy`) can never leak into
the model or the policy weights — it is used **only** by the fairness
guardrail's audit step, exactly as intended.

## 4. Real data, not a curated sample
`src/data_gen.py` produces 4,000 candidates × 120 jobs × ~7,200 realistic
impression-level log rows across 3 tenants, with a genuine
impression→click→application→shortlist funnel (each stage conditioned on
the previous, not independent draws) and historically-biased ground truth
baked in — so the fairness guardrail has something real to catch, not a
toy example. Evaluation is by **held-out job groups** (`GroupShuffleSplit`
in `model.py`) so no candidate/job pair is scored on data it trained on.

## 5. How to run (reproduces every number in `evidence/`)
```
cd src
python data_gen.py      # generates data/logs.pkl etc.
python model.py         # trains model, prints offline eval vs baseline
python demo.py          # Stage E: live config change + guardrail rejection + induced failure

# Automated tests and verification
cd ..
python tests/test_pipeline.py       # guardrail rejection, degraded mode, preview-before-commit
python tests/train_serve_skew.py    # train/serve feature parity check
```
All console output was captured verbatim into `evidence/*.log` — nothing
in this repo is a claim without a matching run log. All paths are relative
to the project root (`config.py` resolves `PROJECT_ROOT` from `__file__`),
so this works on any machine from a fresh clone.

## 6. Definition of Done — status
| Item | Status | Evidence |
|---|---|---|
| Policy layer per tenant, real, demoable | ✅ | `evidence/03_full_demo.log` Steps 1–3 |
| Guardrails block unfair/nonsensical configs | ✅ | `evidence/03_full_demo.log` Step 4, `evidence/04_fairness_guardrail_trigger.log` |
| Admin preview before commit | ✅ | `evidence/03_full_demo.log` Step 2 (before/after top-5, funnel impact, guardrail verdict — all shown pre-commit) |
| Live config change + guardrail rejection demo | ✅ | `evidence/03_full_demo.log` |
| Offline metric vs baseline on held-out data | ✅ | `evidence/02_model_train_eval.log`: nDCG@10 model 0.954 vs baseline 0.897 (+6.4%) |
| Explainable output | ✅ | `serve.py::explain_top_pick`, printed every run |
| Designed degradation when model unavailable | ✅ | `evidence/03_full_demo.log` Step 5 — `degraded_mode=True`, pipeline keeps serving |
| Config versioning + audit log | ✅ | `policy.py::PolicyStore.commit` appends to `data/policy_audit_log.jsonl` |
| Equal-opportunity (TPR parity) guardrail | ✅ | `guardrails.py::check_fairness` — see `evidence/03_full_demo.log`, `evidence/04_fairness_guardrail_trigger.log` |
| LightGBM with GBR fallback | ✅ | `model.py::train_model` — try/except ImportError, see `evidence/02_model_train_eval.log` |
| Train/serve skew check | ✅ | `evidence/06_train_serve_skew_check.log` — bit-for-bit feature match |
| Automated assertion tests | ✅ | `evidence/05_automated_tests.log` — 3/3 PASS (degraded mode, guardrail rejection, preview-before-commit) |
| Portable paths (no hardcoded absolutes) | ✅ | `config.py` resolves `PROJECT_ROOT` from `__file__`; data/ and evidence/ auto-created |

## 7. Pitfalls explicitly addressed
- **Protected-attribute proxy filters:** blocked structurally (never a
  feature) and audited via the four-fifths disparate-impact check
  (`guardrails.check_fairness`), which is shown actively **rejecting** a
  biased config in `evidence/04_fairness_guardrail_trigger.log`.
- **No preview/rollback:** every commit is versioned; preview always runs
  before commit and nothing is live until `PolicyStore.commit()` is called
  explicitly after a passing guardrail check.
- **Offline win never validated online:** `model.py` reports the offline
  nDCG lift explicitly as an *estimate*, not a claim of online lift — the
  README and code comments flag that online A/B validation is the next
  step before wider rollout (out of scope for a single-engineer task demo,
  called out honestly rather than glossed over).
- **Fairness audit as a one-time formality:** the fairness check is wired
  into the commit path itself (`preview.py` → `guardrails.validate_config`),
  not a separate offline notebook — every config change is audited, every time.
- **No model versioning:** `PolicyConfig.version` + append-only
  `policy_audit_log.jsonl` records every committed change with actor and
  timestamp.

## 8. Honest limitations (things a senior reviewer would probe)
- The base model uses LightGBM when available, with an automatic fallback
  to `GradientBoostingRegressor` when `lightgbm` is not installed (see
  `model.py::train_model`, handled via `try/except ImportError`). The
  pointwise approach is explicitly chosen and defensible for this task
  (see §2), but a true listwise learning-to-rank objective (LambdaMART)
  is the natural upgrade and would likely move nDCG further.
- Fairness guardrail now checks **both** the four-fifths selection-rate
  rule **and** equal-opportunity (TPR parity across protected groups).
  Formula: `TPR_g = P(selected=1 | label≥0.5, group=g)` — if
  `min(TPR)/max(TPR) < 0.8`, the config is rejected. See
  `evidence/04_fairness_guardrail_trigger.log`. More advanced group-level
  metrics (calibration parity, predictive parity) are not yet implemented.
- Data is synthetic (generated to match the described log schema), since
  no production log warehouse access exists in this environment — this is
  disclosed throughout rather than presented as real production data.

## 9. Folder structure
```
phase3-task19/
├── README.md                  <- this file
├── requirements.txt
├── experiment_log.md           <- reproducible run log per Stage B/C/D.2
├── data/                       <- generated by data_gen.py / model.py (not hand-curated)
│   ├── candidates.pkl
│   ├── jobs.pkl
│   ├── logs.pkl
│   ├── model.joblib
│   ├── test_scored.pkl
│   └── policy_audit_log.jsonl  <- append-only config audit trail (created on first commit)
├── src/
│   ├── config.py               <- PROJECT_ROOT / DATA_DIR resolved from __file__ (no hardcoded paths)
│   ├── data_gen.py
│   ├── features.py
│   ├── model.py
│   ├── policy.py
│   ├── guardrails.py
│   ├── preview.py
│   ├── serve.py
│   └── demo.py                 <- run this for the live 2-minute demo
├── tests/
│   ├── test_pipeline.py        <- automated assertions: degraded mode, guardrail rejection, preview-before-commit
│   └── train_serve_skew.py     <- verifies train/serve feature parity (bit-for-bit)
└── evidence/                   <- verbatim captured output from real runs
    ├── 01_data_gen.log
    ├── 02_model_train_eval.log
    ├── 03_full_demo.log
    ├── 04_fairness_guardrail_trigger.log
    ├── 05_automated_tests.log
    └── 06_train_serve_skew_check.log
```
