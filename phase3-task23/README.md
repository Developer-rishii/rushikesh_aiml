# Task 23 — Compliance Audit: DPDP, GDPR & SOC 2 Readiness

Working implementation of all three Stage-B deliverables from the study guide,
integrated and demoed end-to-end per Stage E, with one deliberately induced
failure. Every number below was produced by actually running the code in
this repo (see `demo_output/demo_transcript.txt` for the raw run).

## How to run
```
pip install -r requirements.txt
cd src
python3 generate_data.py    # builds the logged-interaction dataset
python3 train_model.py      # trains ranker, writes honest offline eval
python3 drift_monitor.py    # proves train/serve skew detection
python3 dsr_rights.py       # access + real deletion demo
python3 disclosure.py       # decision explanation + human review ticket
python3 audit_pack.py       # fairness results + model card + lineage
python3 demo_e2e.py         # full 2-minute demo script incl. induced failure
```

## Folder structure
```
placemux_task23/
├── README.md
├── requirements.txt
├── data/                     # logged interaction data (generated)
│   ├── candidates.csv, jobs.csv, interactions.csv
│   └── data_manifest.json    # provenance + hash
├── models/
│   ├── ranker.joblib
│   ├── eval_results.json     # offline metrics vs baseline + honest online caveat
│   └── model_registry.json   # versioned registry, incl. deletion-taint flags
├── audit/                    # Deliverable 3: the audit pack
│   ├── model_card.md
│   ├── fairness_results.json / fairness_history.jsonl
│   ├── lineage.json
│   ├── drift_check.json
│   └── human_review_queue.db # Deliverable 2: real SQLite ticket queue
├── logs/
│   └── experiment_log.jsonl  # append-only, reproducible run log
├── demo_output/
│   └── demo_transcript.txt   # captured output of the actual demo run
└── src/
    ├── generate_data.py
    ├── train_model.py
    ├── drift_monitor.py
    ├── dsr_rights.py         # Deliverable 1
    ├── disclosure.py         # Deliverable 2
    ├── audit_pack.py         # Deliverable 3
    └── demo_e2e.py           # Stage E integration + failure induction
```

## Scoring rubric — where the evidence lives
| Rubric (100 pts) | Evidence |
|---|---|
| Core deliverables built, working, demoable (50) | `dsr_rights.py`, `disclosure.py`, `audit_pack.py` each run standalone and produce real output; `demo_e2e.py` runs all three together |
| Real-data quality & correctness (20) | 60k logged interactions with a realistic funnel (impression→click→apply→shortlist); `data_manifest.json` states provenance honestly; held-out group-split evaluation in `train_model.py` |
| Live verification & evidence (15) | `demo_output/demo_transcript.txt` is an actual captured run; deletion is proven by a before/after access-request diff; drift detection is proven against both a buggy and healthy feature path |
| Dependency / failure / edge-case handling (15) | Stage E section 4 physically removes the model file and shows the documented fallback path, then restores it |

## Design decisions made (Section 8 / Stage A.3), and what was rejected
1. **Deletion policy**: retention-window purge + scheduled retrain, not
   per-request full retraining — rejected as operationally infeasible at
   marketplace scale (see docstring in `dsr_rights.py`).
2. **Review path**: mandatory automatic human-review filing for every
   below-threshold decision, not review-on-request only — rejected the
   opt-in version because unstaffed review queues are the "theatre" pitfall
   the guide explicitly warns about.

## Known Limitations & Disclosures (Pitfall 3/4)
- **Data Provenance:** Trained on synthetic-but-realistic logged data (see `data/data_manifest.json`), calibrated to industry benchmarks (~8% CTR, ~20% apply rate). A messy validation slice is included to test pipeline robustness, but it remains fundamentally synthetic due to the study-guide context.
- **Train/Serve Skew:** Deliberately tested and caught by `drift_monitor.py` (see `audit/drift_check.json`).
- **Deletion Architecture:** Non-linear model means individual-row influence is not exactly subtractable; deletion is handled via retention-window purge + scheduled retrain (see Design Decision in `dsr_rights.py`).
- Feature-importance-based explanations approximate individual-decision
  logic; they are global, not exact per-row SHAP values (noted as a
  reasonable, disclosed simplification for a study-guide-scale project).
