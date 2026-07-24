# Hand-off — Task 9

## What the next team needs to know
- **Integration point:** `ExperimentFramework.assign(user_id)` and
  `.route_model(assignment)` in `src/experiment_framework.py` are the two
  calls the Frontend/Backend experiment layer needs to hook into. Assignment
  is pure and stateless (a hash), so it can be called from any service
  instance without a shared session store.
- **Traffic ramp:** currently 5% permanent holdout / 10% treatment / 85%
  control (`HOLDOUT_PCT`, `TREATMENT_PCT` constants). Change the constants,
  not the hashing logic, to ramp the candidate model up.
- **Guardrail thresholds:** `src/guardrails.py::TOLERANCE` and
  `CONSECUTIVE_DAYS_TO_HALT`. These were chosen for this demo's traffic
  volume — retune against real variance before production use (see "Go
  deeper" note on off-policy evaluation for a more rigorous approach to
  significance).
- **Model registry:** `reports/model_registry/` is a local-disk stand-in for
  MLflow (per the study guide's recommended-stack list). Swap
  `src/model_registry.py::register/load_registry` for real MLflow calls
  without touching any other module — it's called from exactly one place in
  `run_all.py`.
- **Known simplification:** the candidate ranker (`GradientBoostingRegressor`,
  pointwise) is a stand-in for a production learning-to-rank model. The
  experiment/guardrail/fairness layers are model-agnostic — any model that
  exposes `.score(df) -> np.ndarray` plugs in unchanged.

## Reproducing this exact run
```bash
pip install -r requirements.txt
python3 data/generate_logs.py
python3 run_all.py
python3 tests/test_verification.py
```
All reports are regenerated fresh — nothing in `reports/` is hand-edited.
