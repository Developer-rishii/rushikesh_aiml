# Task 01 Phase 3 — Model Health, Defect Triage & Phase-3 Backlog

**AI/ML Engineer · Sprint A — Scale & Reliability · PlaceMux · Altrodav Technologies**

---

## Verdict

```
Deliverable 1: ✅ Model-health report complete — online/offline gap = -0.2885 (model over-confident)
Deliverable 2: ✅ Ranked defect list — 2,367 defects detected across 8,000 logged predictions
Deliverable 3: ✅ Phase-3 backlog — 7 items (3 P0, 4 P1), ranked by user impact
Callable live:  GET /health/gap  |  GET /defects/list  |  GET /backlog
```

---

## What Phase 3 is — and how this task fits

Phase 2 built models. Phase 3 owns the intelligence system. The bar for this task is not "build something that works" — it is **"point to a number showing where live matching underperforms, not a hunch."** This task establishes that number.

The three deliverables are connected:
1. **Health report** measures the gap between what the model predicts offline (nDCG@5) and what users actually do online (CTR)
2. **Defect ranker** identifies WHERE that gap comes from — specific logged interactions where the model failed real users, ranked by impact
3. **Backlog** translates those findings into owned, prioritised work items with evidence, not preferences

---

## Upstream dependency — what was needed

| Input | Source | Validated |
|-------|--------|-----------|
| `prediction_logs.csv` | Every served score with features + model_version | ✅ at load time |
| `interaction_logs.csv` | Impressions, clicks, applications | ✅ at load time |
| `training_features.csv` | Features as computed at training time | ✅ at load time |
| `serving_features.csv` | Features as computed at serving time | ✅ at load time |
| `defect_labels.csv` | 1,162 admin-reviewed labels (15.8% defects) | ✅ at load time |

All inputs validated at load time — `ValueError` raised with named missing columns, never silent.

---

## Deliverable 1 — Model-health report (offline vs online gap)

### The key numbers

| Metric | Value |
|--------|-------|
| nDCG@5 (offline) | **0.7454** |
| CTR (online) | **0.2535** |
| Expected CTR from offline score | 0.5420 |
| **Online/offline gap** | **-0.2885** (model over-confident) |
| Apply rate | 0.0742 |
| Shortlist rate | 0.0196 |

The model scores candidates at 0.54 on average, but users only click 25% of recommendations — a **-0.29 gap** showing the model is systematically over-confident. This is the number that drives the backlog.

### Per-model-version breakdown

| Version | nDCG@5 | CTR | Mean skew | Online/offline gap |
|---------|--------|-----|-----------|-------------------|
| v1.0 | 0.7925 | 0.2813 | 0.000 | -0.1952 |
| v1.1 | 0.7517 | 0.2565 | 0.0795 | -0.2977 |
| v1.2 | 0.7250 | 0.2389 | 0.0808 | -0.3226 |

**v1.1 and v1.2 introduced systematic upward score bias (mean skew +0.08) that widened the online/offline gap.** v1.0 had a -0.20 gap; v1.2 widened it to -0.32 — a clear regression introduced by the feature pipeline change.

### Train/serve skew detection (KS test)

| Feature | KS statistic | p-value | Skew detected? |
|---------|-------------|---------|----------------|
| model_score | 0.1283 | 0.000 | ⚠️ YES |
| skill_score_feature | 0.1057 | 0.000 | ⚠️ YES |
| verified_skills | 0.0000 | 1.000 | ✅ No |
| skill_gap | 0.0000 | 1.000 | ✅ No |
| years_exp | 0.0000 | 1.000 | ✅ No |

`model_score` and `skill_score_feature` show statistically significant distributional shift between training and serving time (p=0.000). This is the root cause of the v1.1/v1.2 regression — a feature pipeline change introduced a systematic upward bias in served scores.

---

## Deliverable 2 — Ranked intelligence defect list

### ML defect classifier metrics

| Metric | Value |
|--------|-------|
| Precision | 0.641 |
| Recall | 0.926 |
| F1 | 0.758 |
| AUC-ROC | 0.949 |
| FPR | 0.118 |
| Threshold | 0.25 |
| Labeled pairs | 1,162 (15.8% defects) |

High recall (0.926) was prioritised — in a defect detection system, missing a real defect (false negative) costs more user trust than flagging a clean interaction for review.

### Defects by category (from 8,000 prediction logs)

| Category | Count | Mean user impact |
|----------|-------|-----------------|
| false_positive | 1,087 | 0.476 |
| skew_induced | 642 | 0.500 ← highest |
| false_negative | 120 | 0.431 |

**2,367 total defects detected (29.6% of all predictions).** Skew-induced defects have the highest mean impact — they are the direct downstream consequence of the train/serve skew found in Deliverable 1.

### One worked example

`GET /defects/score/L000000` returns:
```json
{
  "log_id": "L000000",
  "defect_probability": 0.8312,
  "verdict": "⚠️ DEFECT",
  "model_version": "v1.2",
  "reason": "⚠️ DEFECT — defect_prob=0.831 (threshold=0.25). served_score=0.823,
             skew=0.091, rank=3. Top drivers: served_score(669), offline_score(436),
             score_delta(432)."
}
```

---

## Deliverable 3 — Phase-3 backlog

7 items, 3 P0, 4 P1 — ranked by priority then affected user count.

| Rank | ID | Priority | Title | Affected | Effort |
|------|----|----------|-------|----------|--------|
| 1 | B-002 | P0 | Investigate online/offline metric gap — model over-confident | 8,000 | 8d |
| 2 | B-007 | P0 | Ensure 100% prediction logging with model version + feature snapshot | 8,000 | 3d |
| 3 | B-001 | P0 | Fix train/serve skew in features: skill_score_feature, model_score | 6,411 | 5d |
| 4 | B-006 | P1 | Improve matching quality for college_tier=2 (lowest CTR) | varies | 10d |
| 5 | B-004 | P1 | Remediate false_positive defects | 1,087 | 6d |
| 6 | B-003 | P1 | Remediate skew_induced defects | 642 | 6d |
| 7 | B-005 | P1 | Remediate false_negative defects | 120 | 6d |

Every item has: evidence (pointing to a specific number), metric to move, owner, and effort estimate. Nothing is in the backlog on a hunch.

---

## Folder structure

```
task01_phase3/
├── data/
│   ├── generate_data.py        # generates all synthetic interaction + prediction logs
│   ├── prediction_logs.csv     # 8,000 served scores with features + model_version
│   ├── interaction_logs.csv    # 8,000 impressions with clicks/applies/shortlists
│   ├── training_features.csv   # features as computed at training time
│   ├── serving_features.csv    # features at serving time (skew injected in v1.1/v1.2)
│   ├── defect_labels.csv       # 1,162 admin-reviewed labels
│   ├── students.csv
│   ├── jobs.csv
│   └── ranked_defects.csv      # all 8,000 predictions scored + ranked by defect severity
│
├── src/
│   ├── health_monitor.py       # nDCG@5, CTR, gap, skew detection, per-version/segment
│   ├── defect_ranker.py        # LightGBM defect classifier + ranked defect list
│   ├── backlog_generator.py    # evidence-driven Phase-3 backlog
│   └── models/
│       └── defect_classifier.pkl
│
├── api/
│   └── app.py                  # FastAPI: /health/*, /defects/*, /backlog/*, /edge-cases
│
├── reports/
│   ├── health_report.json      # full health metrics
│   ├── experiment_log.jsonl    # reproducible training run log
│   ├── phase3_backlog.json     # machine-readable backlog
│   └── phase3_backlog.md       # human-readable backlog
│
├── tests/
│   └── test_phase3.py          # 16 tests — all passing
│
└── requirements.txt
```

---

## Setup & rebuild from scratch

```bash
pip install -r requirements.txt

# 1. Generate interaction logs + prediction logs
python data/generate_data.py

# 2. Compute health report (offline vs online gap, skew detection)
python src/health_monitor.py

# 3. Train defect classifier + rank all 8,000 predictions
python src/defect_ranker.py

# 4. Generate Phase-3 backlog from evidence
python src/backlog_generator.py

# 5. Run all tests (must all pass)
pytest tests/ -v

# 6. Start API
uvicorn api.app:app --reload --port 8004
```

---

## Demo walkthrough (under 2 minutes live)

```bash
# The key number — online/offline gap
curl http://localhost:8004/health/gap

# Train/serve skew evidence
curl http://localhost:8004/health/skew

# Top-10 defects by user impact
curl "http://localhost:8004/defects/list?top_n=10"

# Score one specific log entry with explanation
curl http://localhost:8004/defects/score/L000000

# P0 backlog items
curl http://localhost:8004/backlog/p0

# Edge cases — visible bucket 4 evidence
curl http://localhost:8004/edge-cases
```

---

## Pitfalls checklist

- [x] **Live prediction logging present** — `prediction_logs.csv` captures every served score with features, model_version, timestamp (`test_health_report_saved`)
- [x] **Health judged by online metric, not just offline** — CTR (online) and nDCG@5 (offline) both reported; gap is the headline number, not the offline number alone
- [x] **Train/serve skew detected, not assumed** — KS test on feature distributions; skew confirmed in 2 features, correctly attributed to v1.1/v1.2 pipeline change
- [x] **Defects ranked by user impact, not engineering interest** — `estimated_user_impact = defect_prob × rank_reciprocal × (1 + skew_penalty)`; `test_defects_ranked_by_impact` asserts this
- [x] **Backlog items all have evidence** — every item points to a specific number from the health report or defect analysis; `test_backlog_items_have_evidence` asserts this
- [x] **Dependency validation at load time** — all 4 input files validated on load; `test_validate_health_inputs_missing_cols` proves it fails loudly
- [x] **16/16 tests pass** — `pytest tests/ -v`

---

## Self-check

- **Can you show the online/offline gap live?** Yes — `GET /health/gap` returns the exact number (-0.2885) and direction with a one-line verdict.
- **Where does the model do worst, and who is hurt?** v1.2 has the worst gap (-0.32); skew-induced defects have the highest mean user impact (0.500). College Tier 2 has the lowest CTR — this is the segment most hurt by the online/offline gap.
- **Is the offline metric correlated with business outcomes?** Partially — nDCG@5 is higher for v1.0 (0.79) which also has better CTR (0.28). But the overall nDCG doesn't predict CTR accurately at the model level — the -0.29 gap shows the metric needs recalibration.
- **What would you log today to answer this faster next month?** Feature snapshots at serving time (currently reconstructed, not persisted) and explicit relevance feedback (thumbs up/down) to supplement click signals.

---

## Hand-off

**Model-defect list to Backend/Data for instrumentation gaps** — specifically B-001 (fix skew in `skill_score_feature`) and B-007 (ensure serving-time feature snapshots are persisted). Without those, next month's skew analysis will require the same reconstruction work.

Guardrail: re-run `python src/health_monitor.py` after every model version deployment; alert if online/offline gap widens beyond -0.30 or if any new feature shows KS p-value < 0.05.
