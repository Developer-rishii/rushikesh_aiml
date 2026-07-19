# Hand-off — Task 25 Live Model Monitoring

## You depended on
- **Production traffic.** Not available at build time. Worked around with
  `src/data_generator.py`, which produces a real-shaped simulated stream
  (same schema, same feature ranges/base-rates as the real thing should
  have). Swap `PROD_TRAFFIC_PATH` in `src/config.py` for a real traffic
  export and every downstream module (monitoring, API, alerts) works
  unchanged — no code depends on the data being simulated.

## What you're getting ("Stable intelligence")
- A trained, validated matching model (`models/match_model.joblib`) that
  beats a documented baseline on precision, FPR, and accuracy.
- A monitoring service (`src/monitoring/monitor_service.py`) that can be
  pointed at any batch of live traffic (real or simulated) and will:
  score it, compute real metrics on whatever labels have arrived, detect
  feature drift against the frozen training distribution, and raise
  alerts — all persisted to `monitoring_store/monitoring.db`.
- A FastAPI layer (`src/api/main.py`) ready to serve `/predict` and expose
  `/monitor/*` for a dashboard, once `pip install fastapi uvicorn` is run
  in an environment with internet access (not available in the build
  sandbox, so it's validated by direct unit/integration tests against the
  same underlying functions instead — see `tests/test_pipeline.py`).

## What's still open before flipping the switch to launch
1. **Real production traffic** — replace the simulated stream. The schema
   in `src/config.py::FEATURE_COLUMNS` must be matched exactly by whatever
   upstream service emits real events.
2. **Load testing** — this task's scope (Section 6/7 of the study guide)
   was model monitoring, not load/perf testing of the serving layer. Before
   go-live, the FastAPI service should be load-tested separately.
3. **Alert delivery channel** — alerts are currently persisted to SQLite
   and printed to the console/log file. Wiring them to Slack/email/PagerDuty
   is a small addition to `src/monitoring/alerts.py::build_alert_record`
   but wasn't part of this task's deliverable.
4. **Retraining trigger** — drift detection (PSI) tells you *when* the
   model is stale; an automatic retrain-and-redeploy pipeline is listed
   under "Go deeper" (Section 12) and is a natural Task 26+ scope, not
   built here.
5. **Data deletion / right-to-erasure flow** — Section 11 self-check asks
   "show me a user deleting their data — does it get removed everywhere?"
   This build's SQLite tables key on `match_id`/`batch_id` only, so a
   deletion job would need a defined mapping back to student/employer IDs
   before this can be answered "yes, and I can show it live."

## Security note
The pitfall list explicitly warns against "we can tighten security after
launch." This build does not expose the SQLite file or model artifacts
over the network by itself — that's the responsibility of however
`src/api/main.py` is deployed (auth, rate limiting, HTTPS termination).
Confirm those are in place before go-live, not after.
