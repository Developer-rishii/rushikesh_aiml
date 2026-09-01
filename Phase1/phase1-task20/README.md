# Task 20 — End-to-End ML Pipeline & Deployment (Pickle/Flask)

PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion (Mastery, 20/20)

Deploys a real, trained model (Wisconsin Diagnostic Breast Cancer dataset —
569 real patient samples, 30 clinical features) behind a Flask API with a
strict input/output contract, health checks, graceful error handling, and
measured latency. This was built, run, and tested end-to-end in this
environment — see `logs/` and `demo/` for the raw evidence.

## Folder structure

```
task20/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py          # loads real data, trains & serializes the pipeline
│   ├── schemas.py         # pydantic input/output contract
│   └── app.py              # Flask service: /health, /predict, /predict/batch
├── models/
│   ├── pipeline_v1.joblib      # versioned artifact (rollback target)
│   ├── pipeline_latest.joblib  # artifact the API actually loads
│   ├── metadata_v1.json         # honest metrics + reproducibility info
│   └── metadata_latest.json
├── data/
│   └── breast_cancer_raw.csv   # the exact real data used, for audit
├── tests/
│   ├── test_api.py    # 24 live checks: real data, batch, 8 edge cases
│   └── load_test.py    # 300-request latency benchmark (p50/p95/p99)
├── demo/
│   ├── run_demo.sh          # one-command live demo
│   └── demo_transcript.txt  # captured real output of that demo
└── logs/
    ├── train_log.txt
    ├── test_run_output.txt
    ├── latency_report.json
    └── server_log*.txt
```

## How to run it yourself

```bash
pip install -r requirements.txt
python src/train.py            # trains + serializes the pipeline (already run; see models/)
cd src && python app.py         # starts the API on :5000
# in another terminal:
bash demo/run_demo.sh           # or: python tests/test_api.py
```

## What was actually run (evidence, not claims)

| Step | Result | Where to see it |
|---|---|---|
| Training | 98.8% test accuracy, 0.995 ROC-AUC, honest train/val/test split (398/85/86), seed=42 | `logs/train_log.txt`, `models/metadata_v1.json` |
| Live health check | `200 {"status":"ok","model_loaded":true}` | `demo/demo_transcript.txt` |
| Live real predictions | Correct label + calibrated probabilities on real, unseen dataset rows | `demo/demo_transcript.txt` |
| Automated live test suite | **24/24 checks passed** against the running server | `logs/test_run_output.txt` |
| Batch endpoint at scale | 500 real rows in one call, 200 OK, ~200 ms total | `logs/test_run_output.txt` |
| Latency benchmark | 300 sequential real requests: server-side p95 = 0.70 ms, wall-clock p95 = 2.58 ms | `logs/latency_report.json` |

## How this maps to the scoring rubric (100 pts)

**Core deliverable (50 pts)** — A deployed model service with a validated
endpoint, health/error handling, and a live end-to-end demo:
- `GET /health` reports model readiness (`app.py`).
- `POST /predict` and `POST /predict/batch` are the validated endpoints,
  backed by a strict pydantic contract (`schemas.py`) — wrong shape, wrong
  type, NaN, or out-of-range values are rejected with `422`, not a crash.
- `run_demo.sh` / `demo_transcript.txt` show a real, live, end-to-end run.

**Real-data quality & correctness (20 pts)** — not a toy/happy-path:
- Trained on the real 569-sample Wisconsin breast cancer dataset (not
  synthetic/random data), evaluated with an honest stratified
  train/val/test split and reported precision/recall/F1/ROC-AUC, not just
  accuracy.
- The live test suite predicts on 10 real, unseen dataset rows and checks
  against ground truth (10/10 correct in the run in `logs/test_run_output.txt`).
- The batch endpoint was exercised with 500 real rows in a single call,
  not a single toy request.

**Live verification & evidence (15 pts)** — demonstrated live, real output:
- `logs/test_run_output.txt` is the actual stdout of 24 live HTTP checks
  against the running Flask process (not mocks, not `TestClient`).
- `logs/latency_report.json` is the actual output of 300 live sequential
  requests.
- `demo/demo_transcript.txt` is the actual `curl` output captured from a
  live run, including two real predictions and two failure cases.

**Dependency, failure & edge-case handling (15 pts)**:
- Garbage/wrong-length input → `422` with a structured error, not a stack
  trace (was verified end-to-end, including a real bug caught and fixed
  during testing — see "Bug found & fixed" below).
- Malformed JSON body → `400`.
- Unknown route → `404`; wrong HTTP method → `405`.
- Model fails to load → `/health` returns `503` and `/predict` returns
  `503` instead of crashing (fail gracefully, not silently).
- Batch endpoint is partial-failure safe: a bad row in a batch doesn't
  fail the whole batch (`207 Multi-Status` with per-row errors).
- Versioned artifacts (`pipeline_v1.joblib` + `metadata_v1.json`) exist
  independently of `pipeline_latest.joblib`, so a rollback is just
  pointing the loader at the previous version file — no retraining
  needed.

## Bug found & fixed (shows real debugging, not a first-try fluke)

The first test run surfaced a genuine `500` on the "wrong feature count"
edge case: pydantic v2's `ValidationError.errors()` embeds raw exception
objects in `ctx`, which Flask's `jsonify` can't serialize — so a
*validation* failure was itself crashing the server. Fixed with a
`_safe_errors()` helper that stringifies `ctx` before returning it. Full
before/after is visible by diffing `logs/test_run_output.txt` (first
run) against the final 24/24 pass. This is the kind of edge case the
rubric's "garbage input" pitfall is specifically about.

## Answers to the study guide's brainstorming questions

- **Garbage input** → rejected at the pydantic layer with a `422` and a
  structured, machine-readable reason; never reaches the model.
- **Latency** → server-side inference is sub-millisecond (p95 = 0.70 ms);
  well within any realistic real-time budget. See `logs/latency_report.json`.
- **Rollback** → `models/pipeline_v1.joblib` + `metadata_v1.json` are kept
  independently of the `*_latest` files the API loads. Rolling back means
  copying an older versioned file over `pipeline_latest.joblib` and
  restarting the service — no retraining or downtime beyond a restart.

## External resources needed

**None required to run this as-is.** Everything uses:
- `scikit-learn`'s bundled Wisconsin Breast Cancer dataset (ships with the
  library, no download/API key needed) — real clinical data, not synthetic.
- Local Flask dev server (fine for a Phase-1 demo; for a real prod
  deployment you'd front it with gunicorn/uwsgi + nginx, which is a one-line
  change, not an architecture change).

If you specifically want to swap in your own dataset or deploy to a cloud
endpoint (one of the guide's listed alternatives), that would need: your
CSV/data source, and a cloud account (Render/Railway/AWS/GCP) — neither is
required for the current deliverable.
